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
Sentiment Analyzer module for VNLP Colab.

This module provides the high-level SentimentAnalyzer API, which uses a
singleton factory to manage the SPUCBiGRUSentimentAnalyzer instance. The
implementation is optimized for Keras 3 and high-performance inference.
"""
import logging
import pickle
from typing import Dict, Any, Optional, Tuple

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow import keras

# Updated imports for package structure
from vnlp_colab.utils_colab import download_resource, get_vnlp_cache_dir
from vnlp_colab.sentiment.sentiment_utils_colab import create_spucbigru_sentiment_model, process_sentiment_input

logger = logging.getLogger(__name__)

# --- Model & Resource Configuration ---
_MODEL_CONFIGS = {
    'SPUCBiGRUSentimentAnalyzer': {
        'weights_prod': ("Sentiment_SPUCBiGRU_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Sentiment_SPUCBiGRU_prod.weights"),
        'weights_eval': ("Sentiment_SPUCBiGRU_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Sentiment_SPUCBiGRU_eval.weights"),
        'word_embedding_matrix': ("SPUTokenized_word_embedding_16k.matrix", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"),
        'spu_tokenizer': ("SPU_word_tokenizer_16k.model", "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/resources/SPU_word_tokenizer_16k.model"),
        'params': {
            'text_max_len': 256,
            'word_embedding_dim': 128,
            'num_rnn_stacks': 3,
            'num_rnn_units': 128,
            'dropout_rate': 0.2,
        }
    }
}

# --- Singleton Caching for Model Instances ---
_MODEL_INSTANCE_CACHE: Dict[str, Any] = {}


class SPUCBiGRUSentimentAnalyzer:
    """
    SentencePiece Unigram Context Bidirectional GRU Sentiment Analyzer.
    Optimized with tf.function for high-performance inference.
    """
    def __init__(self, evaluate: bool = False):
        logger.info(f"Initializing SPUCBiGRUSentimentAnalyzer (evaluate={evaluate})...")
        config = _MODEL_CONFIGS['SPUCBiGRUSentimentAnalyzer']
        self.params = config['params']
        cache_dir = get_vnlp_cache_dir()

        # Download and load resources
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        embedding_path = download_resource(*config['word_embedding_matrix'], cache_dir)
        spu_tokenizer_path = download_resource(*config['spu_tokenizer'], cache_dir)

        self.spu_tokenizer_word = spm.SentencePieceProcessor(model_file=str(spu_tokenizer_path))
        word_embedding_matrix = np.load(embedding_path)
        
        vocab_size = self.spu_tokenizer_word.get_piece_size()
        
        self.model = create_spucbigru_sentiment_model(
            vocab_size=vocab_size,
            word_embedding_matrix=np.zeros_like(word_embedding_matrix),
            **self.params
        )
        
        with open(weights_path, 'rb') as fp:
            trainable_weights = pickle.load(fp)

        # The non-trainable embedding matrix is the first weight
        full_weights = [word_embedding_matrix] + trainable_weights
        self.model.set_weights(full_weights)

        self._initialize_compiled_predict_step()
        logger.info("SPUCBiGRUSentimentAnalyzer initialized successfully.")

    def _initialize_compiled_predict_step(self):
        """Creates a compiled TensorFlow function for a faster forward pass."""
        input_signature = [tf.TensorSpec(shape=(None, self.params['text_max_len']), dtype=tf.int32)]

        @tf.function(input_signature=input_signature)
        def inference_function(input_tensor):
            return self.model(input_tensor, training=False)
        
        self.compiled_predict = inference_function

    def predict_proba(self, text: str) -> float:
        """Predicts the sentiment probability for a given text."""
        processed_input = process_sentiment_input(text, self.spu_tokenizer_word, self.params['text_max_len'])
        prob = self.compiled_predict(tf.constant(processed_input)).numpy()[0][0]
        return float(prob)


class SentimentAnalyzer:
    """
    Main API class for Sentiment Analyzer implementations.
    Uses a singleton factory for efficient model instance management.
    """
    def __init__(self, model: str = 'SPUCBiGRUSentimentAnalyzer', evaluate: bool = False):
        self.available_models = ['SPUCBiGRUSentimentAnalyzer']
        if model not in self.available_models:
            raise ValueError(f"'{model}' is not a valid model. Try one of {self.available_models}")

        cache_key = f"sentiment_{model}_{'eval' if evaluate else 'prod'}"
        if cache_key not in _MODEL_INSTANCE_CACHE:
            logger.info(f"Instance for '{cache_key}' not found. Creating new one.")
            if model == 'SPUCBiGRUSentimentAnalyzer':
                _MODEL_INSTANCE_CACHE[cache_key] = SPUCBiGRUSentimentAnalyzer(evaluate)
        else:
            logger.info(f"Found cached instance for '{cache_key}'.")

        self.instance: SPUCBiGRUSentimentAnalyzer = _MODEL_INSTANCE_CACHE[cache_key]

    def predict(self, text: str) -> int:
        """
        Predicts a discrete sentiment label (1 for positive, 0 for negative).
        """
        prob = self.predict_proba(text)
        return 1 if prob > 0.5 else 0

    def predict_proba(self, text: str) -> float:
        """
        Predicts the probability of a positive sentiment.
        """
        return self.instance.predict_proba(text)

# --- Main Entry Point for Standalone Use ---
def main():
    """Demonstrates and tests the Sentiment Analyzer module."""
    from utils_colab import setup_logging
    setup_logging()
    
    logger.info("--- VNLP Colab Sentiment Analyzer Test Suite ---")
    
    # Test SPUCBiGRUSentimentAnalyzer
    try:
        logger.info("\n1. Testing SPUCBiGRUSentimentAnalyzer...")
        sentiment_analyzer = SentimentAnalyzer()
        sentence_pos = "Bu filmi çok beğendim, harikaydı."
        sentence_neg = "Tam bir zaman kaybı, hiç tavsiye etmiyorum."

        prob_pos = sentiment_analyzer.predict_proba(sentence_pos)
        pred_pos = sentiment_analyzer.predict(sentence_pos)
        logger.info(f"   Input: '{sentence_pos}' -> Proba: {prob_pos:.4f}, Pred: {pred_pos}")
        assert pred_pos == 1 and prob_pos > 0.5

        prob_neg = sentiment_analyzer.predict_proba(sentence_neg)
        pred_neg = sentiment_analyzer.predict(sentence_neg)
        logger.info(f"   Input: '{sentence_neg}' -> Proba: {prob_neg:.4f}, Pred: {pred_neg}")
        assert pred_neg == 0 and prob_neg < 0.5
        
        logger.info("   SPUCBiGRUSentimentAnalyzer test PASSED.")
    except Exception as e:
        logger.error(f"   SPUCBiGRUSentimentAnalyzer test FAILED: {e}", exc_info=True)

    # Test Singleton Caching
    logger.info("\n2. Testing Singleton Caching...")
    import time
    start_time = time.time()
    _ = SentimentAnalyzer()
    end_time = time.time()
    logger.info(f"   Re-initialization took: {end_time - start_time:.4f} seconds.")
    assert (end_time - start_time) < 0.1, "Caching failed, re-initialization was too slow."
    logger.info("   Singleton Caching test PASSED.")

if __name__ == "__main__":
    main()