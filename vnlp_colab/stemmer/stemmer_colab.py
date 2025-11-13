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
from vnlp_colab.utils_colab import download_resource, load_keras_tokenizer, get_vnlp_cache_dir
from vnlp_colab.stemmer.stemmer_utils_colab import create_stemmer_model, process_stemmer_input
from vnlp_colab.stemmer._yildiz_analyzer import get_candidate_generator_instance, capitalize

logger = logging.getLogger(__name__)

# --- Model & Resource Configuration ---
_MODEL_CONFIGS = {
    'StemmerAnalyzer': {
        'weights_prod': ("Stemmer_Shen_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Stemmer_Shen_prod.weights"),
        'weights_eval': ("Stemmer_Shen_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Stemmer_Shen_eval.weights"),
        'char_tokenizer': ("Stemmer_char_tokenizer.json", "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/stemmer_morph_analyzer/resources/Stemmer_char_tokenizer.json"),
        'tag_tokenizer': ("Stemmer_morph_tag_tokenizer.json", "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/stemmer_morph_analyzer/resources/Stemmer_morph_tag_tokenizer.json"),
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
# In vnlp_colab/stemmer/stemmer_colab.py

    def __init__(self, evaluate: bool = False):
        """
        Initializes the model, loads all necessary resources, and compiles the
        prediction function. This is a heavyweight operation managed by the
        singleton factory.
        """
        logger.info(f"Initializing StemmerAnalyzer model (evaluate={evaluate})...")
        config = _MODEL_CONFIGS['StemmerAnalyzer']
        self.params = config['params']
        cache_dir = get_vnlp_cache_dir()

        # --- FIX: Isolate the post-processing parameter ---
        self.capitalize_pnons = self.params.pop('capitalize_pnons', False)

        # --- Download and Load Resources ---
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        char_tokenizer_path = download_resource(*config['char_tokenizer'], cache_dir)
        tag_tokenizer_path = download_resource(*config['tag_tokenizer'], cache_dir)

        self.tokenizer_char = load_keras_tokenizer(char_tokenizer_path)
        self.tokenizer_tag = load_keras_tokenizer(tag_tokenizer_path)
        self.tokenizer_tag.filters = ''
        self.tokenizer_tag.split = ' '

        char_vocab_size = len(self.tokenizer_char.word_index) + 1
        tag_vocab_size = len(self.tokenizer_tag.word_index) + 1

        # --- Build and Load Model ---
        # The **self.params call is now safe because 'capitalize_pnons' has been removed.
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

    def predict(self, tokens: List[str], batch_size: int = 64) -> List[str]:
        """High-level API for Morphological Disambiguation on pre-tokenized text.
        
        Args:
            tokens (list): Input sentence tokens.
            batch_size (int): Number of tokens to process in one model call.

        Returns:
            List[str]: A list of the selected morphological analyses.
        """
        if not tokens:
            return []

        sentence_analyses = [self.candidate_generator.get_analysis_candidates(token) for token in tokens]
        data_for_processing = [(tokens, sentence_analyses)]
        
        x_numpy, _ = process_stemmer_input(
            data_for_processing, self.tokenizer_char, self.tokenizer_tag,
            stem_max_len=self.params['stem_max_len'],
            tag_max_len=self.params['tag_max_len'],
            surface_token_max_len=self.params['surface_token_max_len'],
            sentence_max_len=self.params['sentence_max_len'],
            num_max_analysis=self.params['num_max_analysis']
        )
        
        if len(tokens) <= batch_size:
            probs = self.compiled_predict_step(*x_numpy)
        else:
            all_probs = []
            for i in range(0, len(tokens), batch_size):
                batch_x = tuple(x[i : i + batch_size] for x in x_numpy)
                all_probs.append(self.compiled_predict_step(*batch_x))
            probs = tf.concat(all_probs, axis=0)

        ambig_levels = np.array([len(a) for a in sentence_analyses], dtype=np.int32)
        mask = tf.sequence_mask(ambig_levels, maxlen=self.params['num_max_analysis'], dtype=tf.float32)
        
        predicted_indices = tf.argmax(probs * mask, axis=-1).numpy()

        final_result = []
        for i, analyses in enumerate(sentence_analyses):
            pred_idx = predicted_indices[i]
            if pred_idx < len(analyses):
                root, _, tags = analyses[pred_idx]
                # --- FIX: Use the instance attribute directly ---
                if "Prop" in tags and self.capitalize_pnons:
                    root = capitalize(root)
                analysis_str = "+".join([root] + tags).replace('+DB', '^DB')
                final_result.append(analysis_str)
            else:
                final_result.append(tokens[i] + "+Unknown")

        return final_result


def get_stemmer_analyzer(evaluate: bool = False) -> StemmerAnalyzer:
    """Singleton factory function for the StemmerAnalyzer."""
    cache_key = f"stemmer_{'eval' if evaluate else 'prod'}"
    if cache_key not in _MODEL_INSTANCE_CACHE:
        _MODEL_INSTANCE_CACHE[cache_key] = StemmerAnalyzer(evaluate=evaluate)
    return _MODEL_INSTANCE_CACHE[cache_key]