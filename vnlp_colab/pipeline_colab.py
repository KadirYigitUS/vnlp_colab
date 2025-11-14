# vnlp_colab/pipeline_colab.py
# coding=utf-8
#
# Copyright 2025 VNLP Project Authors.
#
# Licensed under the GNU Affero General Public License, Version 3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/agpl-3.0.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unified NLP Pipeline for VNLP Colab.

This module provides a high-level API to run a sequence of NLP tasks on a
dataset, including PoS tagging, NER, dependency parsing, sentiment analysis,
and morphological analysis. It is architected for high-performance, batched
inference using tf.data.
"""
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Set, Generator, Tuple

import pandas as pd
import tensorflow as tf
from tqdm.notebook import tqdm

# Updated imports for package structure
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
    Orchestrates a full NLP analysis pipeline for a given dataset, optimized
    for high-throughput batch processing.
    """
    def __init__(self, models_to_load: List[str]):
        """
        Initializes the pipeline and loads the required models into memory.

        Args:
            models_to_load (List[str]): Models to init. Format: ['task'] or ['task:model_name'].
                Examples: ['pos', 'ner', 'dep:TreeStackDP', 'stemmer', 'sentiment']
        """
        setup_logging()
        logger.info("Initializing VNLP Pipeline...")
        self.models: Dict[str, Any] = {}
        
        # --- Dependency Resolution ---
        resolved_models: Set[str] = set(models_to_load)
        model_map: Dict[str, str] = {}

        for model_str in models_to_load:
            parts = model_str.split(':')
            task = parts[0]
            model_name = parts[1] if len(parts) > 1 else None
            
            if task == 'dep' and model_name == 'TreeStackDP':
                resolved_models.add('pos:TreeStackPoS')
            if task == 'pos' and model_name == 'TreeStackPoS':
                resolved_models.add('stemmer')

        for model_str in resolved_models:
            parts = model_str.split(':')
            task, model_name = (parts[0], parts[1]) if len(parts) > 1 else (parts[0], None)
            model_map[task] = model_name

        logger.info(f"Resolved model loading order: {list(model_map.keys())}")

        # --- Model Loading with Dependency Injection ---
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
        
        logger.info(f"Pipeline initialized successfully with models: {list(self.models.keys())}")

    def load_from_csv(self, file_path: str, pickle_path: str) -> pd.DataFrame:
        logger.info(f"Loading data from '{file_path}'...")
        df = pd.read_csv(
            file_path, sep='\t', header=None,
            names=['t_code', 'ch_no', 'p_no', 's_no', 'sentence'],
            dtype={'sentence': 'string'}
        )
        logger.info(f"Loaded {len(df)} records. Saving initial pickle to '{pickle_path}'...")
        df.to_pickle(pickle_path)
        return df

    def run_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting preprocessing (using vectorized pandas operations)...")
        df['sentence'] = df['sentence'].str.replace(r'\s+', ' ', regex=True).str.strip().fillna("")
        df.dropna(subset=['sentence'], inplace=True)
        logger.info("Step 1/3: Cleaned 'sentence' column.")
        
        df['no_accents'] = df['sentence'].progress_apply(self.normalizer.remove_accent_marks)
        logger.info("Step 2/3: Created 'no_accents' column.")
        
        df['tokens'] = df['no_accents'].progress_apply(TreebankWordTokenize)
        logger.info("Step 3/3: Created 'tokens' column.")
        return df

    def _dataframe_generator(self, df: pd.DataFrame) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
        """A generator that yields data needed for batch processing."""
        for _, row in df.iterrows():
            yield (
                tf.constant(row['tokens'], dtype=tf.string),
                tf.constant(row['sentence'], dtype=tf.string)
            )

    def process_dataframe(self, df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """
        Processes a DataFrame using a high-performance tf.data batching pipeline.
        """
        logger.info(f"Starting NLP model analysis with batch size {batch_size}...")
        
        # --- tf.data Pipeline Setup ---
        dataset = tf.data.Dataset.from_generator(
            lambda: self._dataframe_generator(df),
            output_signature=(
                tf.TensorSpec(shape=(None,), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.string)
            )
        )
        
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([None], []), # Pad the tokens dimension, not the scalar sentence
            padding_values=("<pad>", "")
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # --- Initialize result storage ---
        all_results = {task: [] for task in self.models.keys()}
        if 'sentiment' in self.models:
            all_results['sentiment_proba'] = []

        # --- Batch Processing Loop ---
        num_batches = (len(df) + batch_size - 1) // batch_size
        for batch_tokens_tf, batch_sentences_tf in tqdm(dataset, total=num_batches, desc="Processing Batches"):
            # Convert tensors back to Python types for model processing
            batch_tokens = [[tok.decode('utf-8') for tok in sent if tok.decode('utf-8') != '<pad>'] for sent in batch_tokens_tf.numpy()]
            batch_sentences = [s.decode('utf-8') for s in batch_sentences_tf.numpy()]
            
            # --- Execute models in dependency order ---
            if 'sentiment' in self.models:
                sentiment_probs = self.models['sentiment'].predict_proba_batch(batch_sentences)
                all_results['sentiment_proba'].extend(sentiment_probs)

            if 'stemmer' in self.models:
                morph_results = self.models['stemmer'].predict_batch(batch_tokens)
                all_results['stemmer'].extend(morph_results)

            if 'pos' in self.models:
                pos_results = self.models['pos'].predict_batch(batch_tokens)
                all_results['pos'].extend(pos_results)
            
            if 'ner' in self.models:
                ner_results = self.models['ner'].predict_batch(batch_sentences, batch_tokens)
                all_results['ner'].extend(ner_results)
            
            if 'dep' in self.models:
                dep_results = self.models['dep'].predict_batch(batch_tokens)
                all_results['dep'].extend(dep_results)

        # --- Assign results back to DataFrame ---
        logger.info("Assigning batch results back to DataFrame...")
        # Use .loc to ensure alignment with the (potentially sorted) index
        df_index = df.index
        
        if 'sentiment_proba' in all_results:
            df.loc[df_index, 'sentiment'] = all_results['sentiment_proba']

        if 'stemmer' in all_results:
            df.loc[df_index, 'morph'] = all_results['stemmer']
            logger.info("Deriving Lemmas from morphological analysis...")
            df['lemma'] = df['morph'].apply(
                lambda morph_list: [m.split("+")[0] for m in morph_list if '+' in m] if isinstance(morph_list, list) else []
            )

        if 'pos' in all_results:
            df.loc[df_index, 'pos_tuples'] = all_results['pos']
            df['pos'] = df['pos_tuples'].apply(lambda tuples: [tag for _, tag in tuples] if isinstance(tuples, list) else [])
        
        if 'ner' in all_results:
            df.loc[df_index, 'ner_tuples'] = all_results['ner']
            df['ner'] = df['ner_tuples'].apply(lambda tuples: [tag for _, tag in tuples] if isinstance(tuples, list) else [])
        
        if 'dep' in all_results:
            df.loc[df_index, 'dep_tuples'] = all_results['dep']
            df['dep'] = df['dep_tuples'].apply(lambda tuples: [(head, label) for _, _, head, label in tuples] if isinstance(tuples, list) else [])

        # Clean up intermediate tuple columns
        cols_to_drop = [col for col in ['pos_tuples', 'ner_tuples', 'dep_tuples'] if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        logger.info("NLP model analysis complete.")
        return df

    def run(self, csv_path: str, output_pickle_path: str, batch_size: int = 32) -> pd.DataFrame:
        """Executes the full pipeline: load, preprocess, analyze, and save."""
        df_initial = self.load_from_csv(csv_path, f"{Path(output_pickle_path).stem}.initial.pkl")
        df_preprocessed = self.run_preprocessing(df_initial)
        
        # --- REFINEMENT: Sort by token length for efficient batching ---
        logger.info("Optimizing batching efficiency by sorting data by sentence length...")
        df_preprocessed['token_len'] = df_preprocessed['tokens'].str.len()
        df_preprocessed.sort_values('token_len', inplace=True)
        df_preprocessed.drop(columns=['token_len'], inplace=True)
        # The original index is preserved, which we'll use to restore order later.
        
        start_time = time.time()
        df_processed = self.process_dataframe(df_preprocessed, batch_size)
        end_time = time.time()
        
        # --- REFINEMENT: Restore original order ---
        logger.info("Restoring original sentence order...")
        df_final = df_processed.sort_index()
        
        duration = end_time - start_time
        rows_per_second = len(df_final) / duration if duration > 0 else float('inf')
        
        logger.info(f"--- Performance Summary ---")
        logger.info(f"Total processing time: {duration:.2f} seconds for {len(df_final)} rows.")
        logger.info(f"Throughput: {rows_per_second:.2f} rows/sec")
        logger.info(f"--------------------------")
        
        logger.info(f"Saving final processed DataFrame to '{output_pickle_path}'...")
        df_final.to_pickle(output_pickle_path)
        logger.info("Pipeline execution finished successfully.")
        return df_final