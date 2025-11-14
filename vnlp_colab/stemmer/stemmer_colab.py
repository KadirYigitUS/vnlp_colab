# vnlp_colab/stemmer/stemmer_colab.py
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
Stemmer and Morphological Analyzer module for VNLP Colab.

This module contains the high-level StemmerAnalyzer API, refactored for
Keras 3, high-performance inference, and a modern developer experience.
"""
import logging
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Updated imports for package structure
from vnlp_colab.utils_colab import download_resource, load_keras_tokenizer, get_vnlp_cache_dir, get_resource_path
from vnlp_colab.stemmer.stemmer_utils_colab import create_stemmer_model, process_stemmer_input
from vnlp_colab.stemmer._yildiz_analyzer import get_candidate_generator_instance, capitalize

logger = logging.getLogger(__name__)

# --- Model & Resource Configuration ---
_MODEL_CONFIGS = {
    'StemmerAnalyzer': {
        'weights_prod': ("Stemmer_Shen_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Stemmer_Shen_prod.weights"),
        'weights_eval': ("Stemmer_Shen_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Stemmer_Shen_eval.weights"),
        # --- MODIFIED: These are now loaded directly from package resources ---
        'char_tokenizer': "Stemmer_char_tokenizer.json",
        'tag_tokenizer': "Stemmer_morph_tag_tokenizer.json",
        'params': {
            'num_max_analysis': 10, 'stem_max_len': 10, 'tag_max_len': 15,
            'sentence_max_len': 40, 'surface_token_max_len': 15,
            'char_embed_size': 32, 'tag_embed_size': 32,
            'stem_num_rnn_units': 128, 'tag_num_rnn_units': 128,
            'num_rnn_stacks': 1, 'dropout': 0.2, 'embed_join_type': 'add',
            'capitalize_pnons': False,
        }
    }
}

# --- Singleton Caching for Model Instances ---
_MODEL_INSTANCE_CACHE: Dict[str, Any] = {}


class StemmerAnalyzer:
    """
    High-level API for Morphological Disambiguation.

    This class uses a singleton factory pattern for efficient instance management.
    The underlying model is an implementation of "The Role of Context in Neural
    Morphological Disambiguation", optimized for Keras 3 and Colab.
    """
    def __init__(self, evaluate: bool = False):
        """
        Initializes the model, loads all necessary resources, and compiles the
        prediction function. This is a heavyweight operation managed by the
        singleton factory.
        """
        logger.info(f"Initializing StemmerAnalyzer model (evaluate={evaluate})...")
        config = _MODEL_CONFIGS['StemmerAnalyzer']
        self.params = config['params'].copy() # Use copy to safely pop
        cache_dir = get_vnlp_cache_dir()

        self.capitalize_pnons = self.params.pop('capitalize_pnons', False)

        # --- MODIFIED: Load local resources using get_resource_path ---
        resource_pkg_path = "vnlp_colab.stemmer.resources"
        char_tokenizer_path = get_resource_path(resource_pkg_path, config['char_tokenizer'])
        tag_tokenizer_path = get_resource_path(resource_pkg_path, config['tag_tokenizer'])
        
        # Download only the heavyweight model weights
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)

        self.tokenizer_char = load_keras_tokenizer(char_tokenizer_path)
        self.tokenizer_tag = load_keras_tokenizer(tag_tokenizer_path)
        self.tokenizer_tag.filters = ''
        self.tokenizer_tag.split = ' '

        char_vocab_size = len(self.tokenizer_char.word_index) + 1
        tag_vocab_size = len(self.tokenizer_tag.word_index) + 1

        # --- Build and Load Model ---
        self.model = create_stemmer_model(
            char_vocab_size=char_vocab_size,
            tag_vocab_size=tag_vocab_size,
            **self.params
        )
        
        with open(weights_path, 'rb') as fp:
            self.model.set_weights(pickle.load(fp))

        self.candidate_generator = get_candidate_generator_instance(case_sensitive=True)
        self._initialize_compiled_predict_step()
        logger.info("StemmerAnalyzer model initialized successfully.")

    def _initialize_compiled_predict_step(self):
        """Creates a compiled TensorFlow function for a faster forward pass."""
        p = self.params
        input_signature = [
            tf.TensorSpec(shape=(None, p['num_max_analysis'], p['stem_max_len']), dtype=tf.int32),
            tf.TensorSpec(shape=(None, p['num_max_analysis'], p['tag_max_len']), dtype=tf.int32),
            tf.TensorSpec(shape=(None, p['sentence_max_len'], p['surface_token_max_len']), dtype=tf.int32),
            tf.TensorSpec(shape=(None, p['sentence_max_len'], p['surface_token_max_len']), dtype=tf.int32),
        ]

        @tf.function(input_signature=input_signature)
        def predict_step(stems, tags, left_ctx, right_ctx):
            return self.model([stems, tags, left_ctx, right_ctx], training=False)
            
        self.compiled_predict_step = predict_step

    def predict(self, tokens: List[str]) -> List[str]:
        """
        High-level API for Morphological Disambiguation on a single tokenized sentence.
        
        Args:
            tokens (list): Input sentence tokens.

        Returns:
            List[str]: A list of the selected morphological analyses.
        """
        # This method now delegates to the batch-processing method for consistency.
        if not tokens:
            return []
        
        results = self.predict_batch([tokens])
        return results[0] if results else []

    def predict_batch(self, batch_of_tokens: List[List[str]]) -> List[List[str]]:
        """
        High-performance API for Morphological Disambiguation on a batch of tokenized sentences.
        
        Args:
            batch_of_tokens (List[List[str]]): A list of tokenized sentences.

        Returns:
            List[List[str]]: A list containing the analysis results for each sentence.
        """
        if not batch_of_tokens:
            return []

        # --- 1. Flatten the batch and prepare for processing ---
        sentence_lengths = [len(tokens) for tokens in batch_of_tokens]
        flat_tokens = [token for sentence in batch_of_tokens for token in sentence]

        if not flat_tokens:
            return [[] for _ in batch_of_tokens]

        # --- 2. Generate candidates and create model inputs for all tokens in one go ---
        data_for_processing = []
        for i, sentence_tokens in enumerate(batch_of_tokens):
            sentence_analyses = [self.candidate_generator.get_analysis_candidates(token) for token in sentence_tokens]
            data_for_processing.append((sentence_tokens, sentence_analyses))

        x_numpy, _ = process_stemmer_input(
            data_for_processing, self.tokenizer_char, self.tokenizer_tag,
            **self.params
        )

        # --- 3. Run model prediction on the entire batch of tokens ---
        probs = self.compiled_predict_step(*x_numpy)

        # --- 4. Decode results and reconstruct the batch structure ---
        flat_sentence_analyses = [analysis for _, sent_analyses in data_for_processing for analysis in sent_analyses]
        ambig_levels = np.array([len(a) for a in flat_sentence_analyses], dtype=np.int32)
        mask = tf.sequence_mask(ambig_levels, maxlen=self.params['num_max_analysis'], dtype=tf.float32)
        
        predicted_indices = tf.argmax(probs * mask, axis=-1).numpy()

        flat_final_result = []
        for i, analyses in enumerate(flat_sentence_analyses):
            pred_idx = predicted_indices[i]
            if pred_idx < len(analyses):
                root, _, tags = analyses[pred_idx]
                if "Prop" in tags and self.capitalize_pnons:
                    root = capitalize(root)
                analysis_str = "+".join([root] + tags).replace('+DB', '^DB')
                flat_final_result.append(analysis_str)
            else:
                flat_final_result.append(flat_tokens[i] + "+Unknown")
        
        # --- 5. Partition the flat results back into per-sentence lists ---
        batched_final_result = []
        current_pos = 0
        for length in sentence_lengths:
            batched_final_result.append(flat_final_result[current_pos : current_pos + length])
            current_pos += length

        return batched_final_result


def get_stemmer_analyzer(evaluate: bool = False) -> StemmerAnalyzer:
    """Singleton factory function for the StemmerAnalyzer."""
    cache_key = f"stemmer_{'eval' if evaluate else 'prod'}"
    if cache_key not in _MODEL_INSTANCE_CACHE:
        _MODEL_INSTANCE_CACHE[cache_key] = StemmerAnalyzer(evaluate=evaluate)
    return _MODEL_INSTANCE_CACHE[cache_key]

# --- Main entry point for standalone testing ---
def main():
    """Demonstrates and tests the Stemmer/Morphological Analyzer module."""
    from vnlp_colab.utils_colab import setup_logging
    setup_logging()
    
    logger.info("--- VNLP Colab Stemmer/Morphological Analyzer Test Suite ---")
    
    # Test single prediction
    try:
        logger.info("\n1. Testing single prediction...")
        stemmer = get_stemmer_analyzer()
        tokens = ["Üniversite", "sınavlarına", "canla", "çalışıyorlardı", "."]
        result = stemmer.predict(tokens)
        logger.info(f"   Input: {tokens}")
        logger.info(f"   Output: {result}")
        assert len(result) == 5
        assert result[1] == "sınav+Noun+A3pl+P3sg+Dat"
        logger.info("   Single prediction test PASSED.")
    except Exception as e:
        logger.error(f"   Single prediction test FAILED: {e}", exc_info=True)

    # Test batch prediction
    try:
        logger.info("\n2. Testing batch prediction...")
        stemmer = get_stemmer_analyzer()
        batch_of_tokens = [
            ["Benim", "adım", "Melikşah", "."],
            ["Vapurla", "Beşiktaş'a", "geçip", "ulaştım", "."],
            [] # Empty list test case
        ]
        batch_result = stemmer.predict_batch(batch_of_tokens)
        logger.info(f"   Batch Input: {batch_of_tokens}")
        logger.info(f"   Batch Output: {batch_result}")
        assert len(batch_result) == 3
        assert len(batch_result[0]) == 4
        assert len(batch_result[1]) == 5
        assert len(batch_result[2]) == 0
        assert batch_result[0][2] == "melikşah+Noun+Prop+A3sg+Pnon+Nom"
        assert batch_result[1][1] == "beşiktaş+Noun+Prop+A3sg+Pnon+Dat"
        logger.info("   Batch prediction test PASSED.")
    except Exception as e:
        logger.error(f"   Batch prediction test FAILED: {e}", exc_info=True)


if __name__ == "__main__":
    main()