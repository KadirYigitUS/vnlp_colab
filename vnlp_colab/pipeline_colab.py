# vnlp_colab/pipeline_colab.py
# coding=utf-8
# Copyright 2025 VNLP Project Authors.
# Licensed under AGPL-3.0

"""
Unified NLP Pipeline for VNLP Colab.

This module provides the high-level orchestration for the NLP pipeline.
It implements the "Explode-Process-Implode" pattern to handle token limitations:
1. Sentences are chunked into 40-token segments (`tokens_40`).
2. Data is exploded so each chunk becomes a processing unit.
3. Batch inference runs on chunks.
4. Results are re-aggregated to the original sentence level.
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Generator, Tuple

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm

from vnlp_colab.utils_colab import setup_logging
from vnlp_colab.tokenizer_colab import TreebankWordTokenize
from vnlp_colab.normalizer.normalizer_colab import Normalizer
from vnlp_colab.pos.pos_colab import PoSTagger
from vnlp_colab.ner.ner_colab import NamedEntityRecognizer
from vnlp_colab.dep.dep_colab import DependencyParser
from vnlp_colab.stemmer.stemmer_colab import get_stemmer_analyzer
from vnlp_colab.sentiment.sentiment_colab import SentimentAnalyzer

logger = logging.getLogger(__name__)
tqdm.pandas()


class VNLPipeline:
    """
    Orchestrates the NLP analysis pipeline.
    """
    def __init__(self, models_to_load: List[str]):
        """
        Args:
            models_to_load: List of keys e.g. ['pos', 'ner', 'dep', 'stemmer', 'sentiment']
        """
        setup_logging()
        logger.info("Initializing VNLP Pipeline...")
        self.models: Dict[str, Any] = {}
        
        # --- Dependency Resolution ---
        resolved_models: Set[str] = set(models_to_load)
        # TreeStack models imply dependencies
        for m in models_to_load:
            if 'TreeStackDP' in m: resolved_models.add('pos:TreeStackPoS')
            if 'TreeStackPoS' in m: resolved_models.add('stemmer')

        model_map = {m.split(':')[0]: (m.split(':')[1] if ':' in m else None) for m in resolved_models}
        logger.info(f"Resolved model loading order: {list(model_map.keys())}")

        # --- Singleton Model Loading ---
        if 'stemmer' in model_map:
            self.models['stemmer'] = get_stemmer_analyzer()
        if 'pos' in model_map:
            self.models['pos'] = PoSTagger(model=(model_map['pos'] or 'SPUContextPoS'))
        if 'dep' in model_map:
            self.models['dep'] = DependencyParser(model=(model_map['dep'] or 'SPUContextDP'))
        if 'ner' in model_map:
            self.models['ner'] = NamedEntityRecognizer(model=(model_map['ner'] or 'SPUContextNER'))
        if 'sentiment' in model_map:
            self.models['sentiment'] = SentimentAnalyzer()
        
        self.normalizer = Normalizer(stemmer_analyzer_instance=self.models.get('stemmer'))
        logger.info("Pipeline initialized.")

    def load_from_csv(self, file_path: str, pickle_path: str) -> pd.DataFrame:
        """Loads tab-separated CSV and saves initial backup."""
        logger.info(f"Loading data from '{file_path}'...")
        df = pd.read_csv(
            file_path, sep='\t', header=None,
            names=['t_code', 'ch_no', 'p_no', 's_no', 'sentence'],
            dtype={'sentence': 'string'}
        )
        logger.info(f"Loaded {len(df)} records. Saving backup to '{pickle_path}'...")
        df.to_pickle(pickle_path)
        return df

    def run_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleaning, normalization, tokenization, and chunking (tokens_40).
        """
        logger.info("Starting preprocessing...")
        
        # 1. Clean
        df['sentence'] = df['sentence'].str.replace(r'\s+', ' ', regex=True).str.strip().fillna("")
        df = df[df['sentence'].str.len() > 0].copy().reset_index(drop=True)
        
        # 2. Normalize
        df['no_accents'] = df['sentence'].progress_apply(self.normalizer.remove_accent_marks)
        
        # 3. Tokenize
        df['tokens'] = df['no_accents'].progress_apply(TreebankWordTokenize)

        # 4. Chunk (tokens_40)
        # Creates a List[List[str]], e.g., [['word1'...'word40'], ['word41'...]]
        def chunk_tokens(tokens):
            if not tokens: return []
            return [tokens[i:i + 40] for i in range(0, len(tokens), 40)]

        df['tokens_40'] = df['tokens'].progress_apply(chunk_tokens)
        
        logger.info("Preprocessing complete. 'tokens_40' column created.")
        return df

    def _chunk_generator(self, df_exploded: pd.DataFrame) -> Generator[tf.Tensor, None, None]:
        """Yields chunks for 40-token models."""
        for tokens in df_exploded['tokens_40']:
            yield tf.constant(tokens, dtype=tf.string)

    def process_dataframe(self, df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """
        Executes models using 'Explode -> Process -> Implode' strategy.
        """
        if df.empty: return df
        logger.info(f"Starting analysis (Batch Size: {batch_size})...")

        # --- 1. SENTIMENT ANALYSIS (Sentence Level) ---
        # Sentiment works on the full sentence, so we run it on the original DF
        if 'sentiment' in self.models:
            logger.info("Running Sentiment Analysis (Full Sentence)...")
            sentences = df['no_accents'].tolist()
            # Process in simple batches
            probs = []
            for i in tqdm(range(0, len(sentences), batch_size), desc="Sentiment"):
                batch = sentences[i : i + batch_size]
                probs.extend(self.models['sentiment'].predict_proba_batch(batch))
            df['sentiment'] = probs

        # --- 2. PREPARE FOR 40-TOKEN MODELS (Chunk Level) ---
        # Explode tokens_40 so each row is a single chunk <= 40 tokens
        # We preserve the original index to re-assemble later
        df_exploded = df.explode('tokens_40').reset_index().rename(columns={'index': 'orig_idx'})
        
        # Filter out potential NaNs from empty token lists
        df_exploded = df_exploded[df_exploded['tokens_40'].notna()]
        
        if df_exploded.empty:
            logger.warning("No tokens found for analysis.")
            return df

        # Create Dataset for Chunks
        dataset = tf.data.Dataset.from_generator(
            lambda: self._chunk_generator(df_exploded),
            output_signature=tf.TensorSpec(shape=(None,), dtype=tf.string)
        )
        dataset = dataset.padded_batch(
            batch_size, 
            padded_shapes=([None]), 
            padding_values=b"<pad>"
        ).prefetch(tf.data.AUTOTUNE)

        # Storage for chunk results
        results_map = {k: [] for k in ['stemmer', 'pos', 'ner', 'dep'] if k in self.models}

        # --- 3. BATCH INFERENCE ---
        num_batches = (len(df_exploded) + batch_size - 1) // batch_size
        
        for batch_tf in tqdm(dataset, total=num_batches, desc="Processing Chunks"):
            # Decode tensors to lists of strings
            batch_tokens = [
                [t.decode('utf-8') for t in sent if t.decode('utf-8') != '<pad>'] 
                for sent in batch_tf.numpy()
            ]
            
            # Since these are chunks, 'sentences' for NER are just joined tokens
            batch_sentences_approx = [" ".join(t) for t in batch_tokens]

            if 'stemmer' in self.models:
                results_map['stemmer'].extend(self.models['stemmer'].predict_batch(batch_tokens))
            
            if 'pos' in self.models:
                results_map['pos'].extend(self.models['pos'].predict_batch(batch_tokens))
                
            if 'ner' in self.models:
                results_map['ner'].extend(self.models['ner'].predict_batch(batch_sentences_approx, batch_tokens))
                
            if 'dep' in self.models:
                # Dependency Parser now receives safe 40-token batches
                results_map['dep'].extend(self.models['dep'].predict_batch(batch_tokens))

        # --- 4. ASSIGN RESULTS TO EXPLODED DF ---
        if 'stemmer' in results_map:
            df_exploded['morph'] = results_map['stemmer']
        if 'pos' in results_map:
            df_exploded['pos'] = results_map['pos']
        if 'ner' in results_map:
            df_exploded['ner'] = results_map['ner']
        if 'dep' in results_map:
            df_exploded['dep'] = results_map['dep']

        # --- 5. IMPLODE (AGGREGATE) BACK TO ORIGINAL ROWS ---
        logger.info("Aggregating chunk results...")
        
        # We group by the original index and sum the lists (concatenation)
        agg_funcs = {}
        if 'stemmer' in self.models: agg_funcs['morph'] = 'sum'
        if 'pos' in self.models: agg_funcs['pos'] = 'sum'
        if 'ner' in self.models: agg_funcs['ner'] = 'sum'
        if 'dep' in self.models: agg_funcs['dep'] = 'sum'

        if agg_funcs:
            # GroupBy sums the lists: [chunk1_res] + [chunk2_res] -> [full_res]
            df_imploded = df_exploded.groupby('orig_idx').agg(agg_funcs)
            
            # Merge back into original DF
            df = df.join(df_imploded)

        # --- 6. POST-PROCESSING (Lemmas & formatting) ---
        if 'stemmer' in self.models and 'morph' in df.columns:
            logger.info("Deriving Lemmas...")
            df['lemma'] = df['morph'].apply(
                lambda x: [m.split("+")[0] for m in x if isinstance(m, str) and '+' in m] 
                if isinstance(x, list) else []
            )

        # Flatten tuples for cleaner output (Pos/Dep/Ner usually return tuples)
        if 'pos' in df.columns:
            # POS returns (token, tag) -> extract tag
            df['pos'] = df['pos'].apply(lambda x: [item[1] for item in x] if isinstance(x, list) else [])
        
        if 'ner' in df.columns:
            # NER returns (token, tag) -> extract tag
            df['ner'] = df['ner'].apply(lambda x: [item[1] for item in x] if isinstance(x, list) else [])
            
        if 'dep' in df.columns:
            # DEP returns (id, word, head, rel) -> extract (head, rel)
            # CAUTION: Head indices in chunks (0-40) need adjustment if we wanted global indices.
            # However, standard DP behavior on chunks implies local heads. 
            # We keep local heads for now as global re-linking is complex without a graph parser.
            df['dep'] = df['dep'].apply(lambda x: [(item[2], item[3]) for item in x] if isinstance(x, list) else [])

        return df

    def run(self, csv_path: str, output_pickle_path: str, batch_size: int = 32) -> pd.DataFrame:
        df_initial = self.load_from_csv(csv_path, f"{Path(output_pickle_path).stem}.initial.pkl")
        df_prep = self.run_preprocessing(df_initial)
        
        # Sort by length of 'tokens' (flattened count) for efficiency
        df_prep['len'] = df_prep['tokens'].str.len()
        df_prep.sort_values('len', inplace=True)
        
        start = time.time()
        df_final = self.process_dataframe(df_prep, batch_size)
        duration = time.time() - start
        
        # Restore order
        df_final.sort_index(inplace=True)
        df_final.drop(columns=['len'], errors='ignore', inplace=True)
        
        print(f"Processing finished in {duration:.2f}s ({len(df_final)/duration:.1f} rows/s)")
        
        logger.info(f"Saving to {output_pickle_path}")
        df_final.to_pickle(output_pickle_path)
        return df_final