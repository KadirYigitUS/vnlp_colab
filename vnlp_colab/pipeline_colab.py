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
and morphological analysis. It supports model selection and dependency chains.
"""
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Set

import pandas as pd
from tqdm.notebook import tqdm

# Updated imports for package structure
from vnlp_colab.utils_colab import setup_logging
from vnlp_colab.tokenizer_colab import TreebankWordTokenize
from vnlp_colab.normalizer.normalizer_colab import Normalizer
from vnlp_colab.pos.pos_colab import PoSTagger
from vnlp_colab.ner.ner_colab import NamedEntityRecognizer
from vnlp_colab.dep.dep_colab import DependencyParser
from vnlp_colab.stemmer.stemmer_colab import StemmerAnalyzer, get_stemmer_analyzer
from vnlp_colab.sentiment.sentiment_colab import SentimentAnalyzer

logger = logging.getLogger(__name__)
tqdm.pandas()


class VNLPipeline:
    """
    Orchestrates a full NLP analysis pipeline for a given dataset.
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

        # Create a clean map for loading        
        for model_str in resolved_models:
            parts = model_str.split(':')
            task, model_name = (parts[0], parts[1]) if len(parts) > 1 else (parts[0], None)
            model_map[task] = model_name

        logger.info(f"Resolved model loading order: {model_map}")

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
        logger.info("Starting preprocessing...")
        df['sentence'] = df['sentence'].progress_apply(
            lambda s: re.sub(r'\s+', ' ', s).strip() if isinstance(s, str) else ""
        )
        logger.info("Step 1/4: Cleaned 'sentence' column.")
        df['no_accents'] = df['sentence'].progress_apply(self.normalizer.remove_accent_marks)
        logger.info("Step 2/4: Created 'no_accents' column.")
        df['tokens'] = df['no_accents'].progress_apply(TreebankWordTokenize)
        logger.info("Step 3/4: Created 'tokens' column.")
        df['tokens_40'] = df['tokens'].progress_apply(
            lambda tokens: [tokens[i:i + 40] for i in range(0, len(tokens), 40)] if tokens else []
        )
        logger.info("Step 4/4: Created 'tokens_40' column for Dependency Parser.")
        return df

    def run_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting NLP model analysis in dependency order...")

        # --- Models are run in a fixed order to satisfy dependencies ---
        if 'sentiment' in self.models:
            logger.info("Running Sentiment Analysis...")
            df['sentiment'] = df['no_accents'].progress_apply(self.models['sentiment'].predict_proba)

        if 'stemmer' in self.models:
            logger.info("Running Morphological Analysis...")
            df['morph'] = df['tokens'].progress_apply(self.models['stemmer'].predict)
            logger.info("Deriving Lemmas...")
            df['lemma'] = df['morph'].progress_apply(
                lambda morph_list: [m.split("+")[0] for m in morph_list if '+' in m]
            )

        if 'pos' in self.models:
            logger.info(f"Running Part-of-Speech Tagging using {self.models['pos'].instance.__class__.__name__}...")
            df['pos_tuples'] = df['tokens'].progress_apply(self.models['pos'].predict)
            df['pos'] = df['pos_tuples'].progress_apply(lambda tuples: [tag for _, tag in tuples])
        
        if 'ner' in self.models:
            logger.info(f"Running Named Entity Recognition using {self.models['ner'].instance.__class__.__name__}...")
            df['ner'] = df.progress_apply(
                lambda row: [tag for _, tag in self.models['ner'].predict(row['no_accents'], row['tokens'])],
                axis=1
            )
        
        if 'dep' in self.models:
            logger.info(f"Running Dependency Parsing using {self.models['dep'].instance.__class__.__name__}...")
            def parse_sentence_batches(row):
                full_result = []
                pos_tuples = row.get('pos_tuples', [])
                if not pos_tuples:
                    pos_tuples = [(token, "X") for token in row['tokens']]

                token_counter = 0
                for batch_tokens in row['tokens_40']:
                    if not batch_tokens: continue
                    batch_pos_tuples = pos_tuples[token_counter : token_counter + len(batch_tokens)]
                    token_counter += len(batch_tokens)
                    batch_dp_result = self.models['dep'].predict(tokens=batch_tokens, pos_result=batch_pos_tuples)
                    simplified_batch = [(head, label) for _, _, head, label in batch_dp_result]
                    full_result.extend(simplified_batch)
                return full_result
            
            df['dep'] = df.progress_apply(parse_sentence_batches, axis=1)

        # Clean up intermediate columns        
        if 'pos_tuples' in df.columns:
            df = df.drop(columns=['pos_tuples'])

        logger.info("NLP model analysis complete.")
        return df

    def run(self, csv_path: str, output_pickle_path: str) -> pd.DataFrame:
        """Executes the full pipeline: load, preprocess, analyze, and save."""
        df_initial = self.load_from_csv(csv_path, f"{Path(output_pickle_path).stem}.initial.pkl")
        df_preprocessed = self.run_preprocessing(df_initial)
        df_final = self.run_analysis(df_preprocessed)
        
        logger.info(f"Saving final processed DataFrame to '{output_pickle_path}'...")
        df_final.to_pickle(output_pickle_path)
        logger.info("Pipeline execution finished successfully.")
        return df_final

if __name__ == "__main__":
    logger.info("--- VNLP Colab Pipeline Standalone Test ---")
    
    dummy_data = (
        "novel01\t1\t1\t1\tBu film harikaydı, çok beğendim.\n"
        "novel01\t1\t1\t2\tBenim adım Melikşah ve İstanbul'da yaşıyorum.\n"
        "novel01\t1\t2\t1\tZamanımı boşa harcadığımı düşünüyorum.\n"
    )
    csv_path = Path("/content/dummy_input.csv")
    csv_path.write_text(dummy_data, encoding='utf-8')
    logger.info(f"Created dummy data at '{csv_path}'")

    # --- Test 1: Full Pipeline (SPUContext Defaults) ---
    logger.info("\n--- Running Test 1: Full SPUContext Pipeline ---")
    spu_models = ['pos', 'ner', 'dep', 'stemmer', 'sentiment']
    spu_pipeline = VNLPipeline(models_to_load=spu_models)
    spu_output_path = "/content/spu_output.pkl"
    spu_df = spu_pipeline.run(csv_path=str(csv_path), output_pickle_path=spu_output_path)

    print("\n--- SPUContext Pipeline Final DataFrame ---")
    pd.set_option('display.max_columns', None); pd.set_option('display.expand_frame_repr', False)
    print(spu_df.head())

    # --- Test 2: TreeStack Dependency Pipeline ---
    logger.info("\n--- Running Test 2: TreeStack Pipeline ---")
    treestack_models = ['dep:TreeStackDP', 'ner', 'sentiment']
    tree_pipeline = VNLPipeline(models_to_load=treestack_models)
    tree_output_path = "/content/treestack_output.pkl"
    tree_df = tree_pipeline.run(csv_path=str(csv_path), output_pickle_path=tree_output_path)
    
    print("\n--- TreeStack Pipeline Final DataFrame ---")
    print(tree_df.head())