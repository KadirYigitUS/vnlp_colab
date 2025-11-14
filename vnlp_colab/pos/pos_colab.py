# vnlp_colab/pos/pos_colab.py
# coding=utf-8
#
# Copyright 2025 VNLP Project Authors
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
Part-of-Speech (PoS) Tagger module for VNLP Colab.

This module contains the high-level PoSTagger API, which uses a singleton
factory to efficiently manage and serve different PoS tagging models, including
SPUContextPoS and the dependency-aware TreeStackPoS.
"""
import logging
import pickle
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow import keras

# Updated imports for package structure
from vnlp_colab.utils_colab import download_resource, load_keras_tokenizer, get_vnlp_cache_dir, get_resource_path
from vnlp_colab.pos.pos_utils_colab import create_spucontext_pos_model, process_pos_input
from vnlp_colab.pos.pos_treestack_utils_colab import create_treestack_pos_model, process_treestack_pos_input
from vnlp_colab.stemmer.stemmer_colab import StemmerAnalyzer, get_stemmer_analyzer
from vnlp_colab.tokenizer_colab import TreebankWordTokenize


logger = logging.getLogger(__name__)

# --- Model & Resource Configuration ---
_MODEL_CONFIGS = {
    'SPUContextPoS': {
        'weights_prod': ("PoS_SPUContext_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_SPUContext_prod.weights"),
        'weights_eval': ("PoS_SPUContext_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_SPUContext_eval.weights"),
        'word_embedding_matrix': ("SPUTokenized_word_embedding_16k.matrix", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"),
        'spu_tokenizer': "SPU_word_tokenizer_16k.model",
        'label_tokenizer': "PoS_label_tokenizer.json",
        'params': { 'word_embedding_dim': 128, 'num_rnn_stacks': 1, 'rnn_units_multiplier': 1, 'fc_units_multiplier': (2, 1), 'dropout': 0.2 }
    },
    'TreeStackPoS': {
        'weights_prod': ("PoS_TreeStack_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_TreeStack_prod.weights"),
        'weights_eval': ("PoS_TreeStack_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_TreeStack_eval.weights"),
        'word_embedding_matrix': ("TBWTokenized_word_embedding.matrix", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/TBWTokenized_word_embedding.matrix"),
        'word_tokenizer': "TB_word_tokenizer.json",
        'morph_tag_tokenizer': "Stemmer_morph_tag_tokenizer.json",
        'pos_label_tokenizer': "PoS_label_tokenizer.json",
        'params': { 'word_embedding_vector_size': 128, 'num_rnn_stacks': 2, 'tag_num_rnn_units': 128, 'lc_num_rnn_units': 256, 'rc_num_rnn_units': 256, 'fc_units_multipliers': (2, 1), 'word_form': 'whole', 'dropout': 0.2, 'sentence_max_len': 40, 'tag_max_len': 15 }
    }
}

# --- Singleton Caching for Model Instances ---
_MODEL_INSTANCE_CACHE: Dict[str, Any] = {}


class SPUContextPoS:
    """
    SentencePiece Unigram Context Part-of-Speech Tagger.
    Optimized with tf.function for high-performance inference.
    """
    def __init__(self, evaluate: bool = False):
        logger.info(f"Initializing SPUContextPoS model (evaluate={evaluate})...")
        config = _MODEL_CONFIGS['SPUContextPoS']
        params = config['params']
        cache_dir = get_vnlp_cache_dir()

        spu_tokenizer_path = get_resource_path("vnlp_colab.resources", config['spu_tokenizer'])
        label_tokenizer_path = get_resource_path("vnlp_colab.pos.resources", config['label_tokenizer'])
        
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        embedding_matrix_path = download_resource(*config['word_embedding_matrix'], cache_dir)

        self.spu_tokenizer_word = spm.SentencePieceProcessor(model_file=str(spu_tokenizer_path))
        self.tokenizer_label = load_keras_tokenizer(label_tokenizer_path)
        self._label_index_word = {i: w for w, i in self.tokenizer_label.word_index.items()}
        word_embedding_matrix = np.load(embedding_matrix_path)

        num_rnn_units = params['word_embedding_dim'] * params['rnn_units_multiplier']
        
        self.model = create_spucontext_pos_model(
            vocab_size=self.spu_tokenizer_word.get_piece_size(),
            pos_vocab_size=len(self.tokenizer_label.word_index),
            word_embedding_dim=params['word_embedding_dim'],
            word_embedding_matrix=np.zeros_like(word_embedding_matrix),
            num_rnn_units=num_rnn_units,
            num_rnn_stacks=params['num_rnn_stacks'],
            fc_units_multiplier=params['fc_units_multiplier'],
            dropout=params['dropout']
        )

        with open(weights_path, 'rb') as fp:
            model_weights = pickle.load(fp)
        
        self.model.set_weights([word_embedding_matrix] + model_weights)
        self._initialize_compiled_predict_step()
        logger.info("SPUContextPoS model initialized successfully.")

    def _initialize_compiled_predict_step(self):
        pos_vocab_size = len(self.tokenizer_label.word_index)
        input_signature = [
            tf.TensorSpec(shape=(None, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 40, pos_vocab_size + 1), dtype=tf.float32),
        ]

        @tf.function(input_signature=input_signature)
        def predict_step(word, left_ctx, right_ctx, lc_pos_history):
            return self.model([word, left_ctx, right_ctx, lc_pos_history], training=False)
        self.compiled_predict_step = predict_step

    def predict(self, tokens: List[str]) -> List[Tuple[str, str]]:
        if not tokens:
            return []
        return self.predict_batch([tokens])[0]

    def predict_batch(self, batch_of_tokens: List[List[str]]) -> List[List[Tuple[str, str]]]:
        if not any(batch_of_tokens):
            return [[] for _ in batch_of_tokens]

        batch_size = len(batch_of_tokens)
        max_len = max(len(s) for s in batch_of_tokens) if batch_of_tokens else 0
        batch_int_preds = [[] for _ in range(batch_size)]

        for t in range(max_len):
            active_indices, step_inputs = [], [], 
            word_batch, left_ctx_batch, right_ctx_batch, lc_pos_history_batch = [], [], [], []
            
            for i in range(batch_size):
                if t < len(batch_of_tokens[i]):
                    active_indices.append(i)
                    inputs_np = process_pos_input(
                        t, batch_of_tokens[i], self.spu_tokenizer_word, 
                        self.tokenizer_label, batch_int_preds[i]
                    )
                    word_batch.append(inputs_np[0])
                    left_ctx_batch.append(inputs_np[1])
                    right_ctx_batch.append(inputs_np[2])
                    lc_pos_history_batch.append(inputs_np[3])

            if not active_indices:
                break

            inputs_tf = [
                tf.convert_to_tensor(np.vstack(word_batch)),
                tf.convert_to_tensor(np.vstack(left_ctx_batch)),
                tf.convert_to_tensor(np.vstack(right_ctx_batch)),
                tf.convert_to_tensor(np.vstack(lc_pos_history_batch)),
            ]
            
            logits = self.compiled_predict_step(*inputs_tf).numpy()
            step_preds = np.argmax(logits, axis=-1)

            for i, pred in enumerate(step_preds):
                original_index = active_indices[i]
                batch_int_preds[original_index].append(pred)

        final_results = []
        for i in range(batch_size):
            labels = [self._label_index_word.get(p, 'UNK') for p in batch_int_preds[i]]
            final_results.append(list(zip(batch_of_tokens[i], labels)))
            
        return final_results


class TreeStackPoS:
    """Tree-stack Part of Speech Tagger class."""
    def __init__(self, stemmer_analyzer: StemmerAnalyzer, evaluate: bool = False):
        logger.info(f"Initializing TreeStackPoS model (evaluate={evaluate})...")
        self.stemmer_analyzer = stemmer_analyzer
        config = _MODEL_CONFIGS['TreeStackPoS']
        self.params = config['params']
        cache_dir = get_vnlp_cache_dir()

        word_tok_path = get_resource_path("vnlp_colab.resources", config['word_tokenizer'])
        morph_tok_path = get_resource_path("vnlp_colab.stemmer.resources", config['morph_tag_tokenizer'])
        pos_tok_path = get_resource_path("vnlp_colab.pos.resources", config['pos_label_tokenizer'])

        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        embedding_path = download_resource(*config['word_embedding_matrix'], cache_dir)

        self.tokenizer_word = load_keras_tokenizer(word_tok_path)
        self.tokenizer_morph_tag = load_keras_tokenizer(morph_tok_path)
        self.tokenizer_pos_label = load_keras_tokenizer(pos_tok_path)
        self._pos_index_word = {i: w for w, i in self.tokenizer_pos_label.word_index.items()}
        
        word_embedding_matrix = np.load(embedding_path)
        tag_embedding_matrix = self.stemmer_analyzer.model.layers[5].weights[0].numpy()
        
        self.model = create_treestack_pos_model(
            word_embedding_vocab_size=len(self.tokenizer_word.word_index) + 1,
            pos_vocab_size=len(self.tokenizer_pos_label.word_index),
            word_embedding_matrix=np.zeros_like(word_embedding_matrix),
            tag_embedding_matrix=tag_embedding_matrix,
            **self.params
        )
        with open(weights_path, 'rb') as fp: model_weights = pickle.load(fp)
        self.model.set_weights([model_weights[0], word_embedding_matrix] + model_weights[1:])
        logger.info("TreeStackPoS model initialized successfully.")

    def predict(self, tokens: List[str]) -> List[Tuple[str, str]]:
        if not tokens:
            return []
        return self.predict_batch([tokens])[0]
        
    def predict_batch(self, batch_of_tokens: List[List[str]]) -> List[List[Tuple[str, str]]]:
        if not any(batch_of_tokens):
            return [[] for _ in batch_of_tokens]

        batch_size = len(batch_of_tokens)
        max_len = max(len(s) for s in batch_of_tokens) if batch_of_tokens else 0
        batch_int_preds = [[] for _ in range(batch_size)]
        
        # Pre-calculate morphological analyses for the entire batch
        batch_analyses = self.stemmer_analyzer.predict_batch(batch_of_tokens)

        for t in range(max_len):
            active_indices = []
            step_inputs_list = []
            
            for i in range(batch_size):
                if t < len(batch_of_tokens[i]):
                    active_indices.append(i)
                    inputs_np = process_treestack_pos_input(
                        batch_of_tokens[i], batch_analyses[i], batch_int_preds[i],
                        self.tokenizer_word, self.tokenizer_morph_tag, self.tokenizer_pos_label,
                        self.params['word_form'], self.params['sentence_max_len'], self.params['tag_max_len']
                    )
                    step_inputs_list.append(inputs_np)

            if not active_indices:
                break
            
            # Stack the inputs for the current time step
            stacked_inputs = [np.vstack([inputs[i] for inputs in step_inputs_list]) for i in range(len(step_inputs_list[0]))]
            
            logits = self.model(stacked_inputs, training=False).numpy()
            step_preds = np.argmax(logits, axis=-1)

            for i, pred in enumerate(step_preds):
                original_index = active_indices[i]
                batch_int_preds[original_index].append(pred)

        final_results = []
        for i in range(batch_size):
            labels = [self._pos_index_word.get(p, 'UNK') for p in batch_int_preds[i]]
            final_results.append(list(zip(batch_of_tokens[i], labels)))
            
        return final_results


class PoSTagger:
    """Main API class for Part-of-Speech Tagger implementations."""
    def __init__(self, model: str = 'SPUContextPoS', evaluate: bool = False):
        self.available_models = ['SPUContextPoS', 'TreeStackPoS']
        if model not in self.available_models:
            raise ValueError(f"'{model}' is not a valid model. Try one of {self.available_models}")

        cache_key = f"pos_{model}_{'eval' if evaluate else 'prod'}"
        if cache_key not in _MODEL_INSTANCE_CACHE:
            logger.info(f"Instance for '{cache_key}' not found. Creating new one.")
            if model == 'SPUContextPoS':
                instance = SPUContextPoS(evaluate)
            elif model == 'TreeStackPoS':
                stemmer_instance = get_stemmer_analyzer(evaluate)
                instance = TreeStackPoS(stemmer_instance, evaluate)
            _MODEL_INSTANCE_CACHE[cache_key] = instance
        else:
            logger.info(f"Found cached instance for '{cache_key}'.")
        self.instance: Union[SPUContextPoS, TreeStackPoS] = _MODEL_INSTANCE_CACHE[cache_key]

    def predict(self, tokens: List[str]) -> List[Tuple[str, str]]:
        return self.instance.predict(tokens)

    def predict_batch(self, batch_of_tokens: List[List[str]]) -> List[List[Tuple[str, str]]]:
        return self.instance.predict_batch(batch_of_tokens)


def main():
    """Demonstrates and tests the PoS Tagger module."""
    from vnlp_colab.utils_colab import setup_logging
    setup_logging()
    logger.info("--- VNLP Colab PoS Tagger Test Suite ---")
    
    # --- SPUContextPoS Tests ---
    try:
        logger.info("\n1. Testing SPUContextPoS...")
        pos_spu = PoSTagger(model='SPUContextPoS')
        
        # Single prediction test
        tokens1 = TreebankWordTokenize("Vapurla Beşiktaş'a geçip yürüyerek Maçka Parkı'na ulaştım.")
        result_spu1 = pos_spu.predict(tokens1)
        logger.info(f"   SPUContextPoS Single Output: {result_spu1}")
        assert len(result_spu1) == len(tokens1) and result_spu1[1][1] == 'PROPN'
        
        # Batch prediction test
        batch1 = [
            TreebankWordTokenize("Benim adım Melikşah."),
            TreebankWordTokenize("İstanbul'da yaşıyorum.")
        ]
        batch_res1 = pos_spu.predict_batch(batch1)
        logger.info(f"   SPUContextPoS Batch Output: {batch_res1}")
        assert len(batch_res1) == 2
        assert batch_res1[0][2][1] == 'PROPN' and batch_res1[1][0][1] == 'PROPN'
        logger.info("   SPUContextPoS test PASSED.")
    except Exception as e:
        logger.error(f"   SPUContextPoS test FAILED: {e}", exc_info=True)

    # --- TreeStackPoS Tests ---
    try:
        logger.info("\n2. Testing TreeStackPoS...")
        pos_tree = PoSTagger(model='TreeStackPoS')

        # Single prediction test
        tokens2 = TreebankWordTokenize("Vapurla Beşiktaş'a geçip yürüyerek Maçka Parkı'na ulaştım.")
        result_tree2 = pos_tree.predict(tokens2)
        logger.info(f"   TreeStackPoS Single Output: {result_tree2}")
        assert len(result_tree2) == len(tokens2) and result_tree2[1][1] == 'PROPN'
        
        # Batch prediction test
        batch2 = [
            TreebankWordTokenize("Benim adım Melikşah."),
            TreebankWordTokenize("İstanbul'da yaşıyorum."),
            [] # Empty list test case
        ]
        batch_res2 = pos_tree.predict_batch(batch2)
        logger.info(f"   TreeStackPoS Batch Output: {batch_res2}")
        assert len(batch_res2) == 3
        assert len(batch_res2[2]) == 0
        assert batch_res2[0][2][1] == 'PROPN' and batch_res2[1][0][1] == 'PROPN'
        logger.info("   TreeStackPoS test PASSED.")
    except Exception as e:
        logger.error(f"   TreeStackPoS test FAILED: {e}", exc_info=True)
    
    # --- Singleton Caching Test ---
    logger.info("\n3. Testing Singleton Caching...")
    import time
    start_time = time.time()
    _ = PoSTagger(model='SPUContextPoS')
    end_time = time.time()
    logger.info(f"   Re-initialization took: {end_time - start_time:.4f} seconds.")
    assert (end_time - start_time) < 0.1
    logger.info("   Singleton Caching test PASSED.")

if __name__ == "__main__":
    main()