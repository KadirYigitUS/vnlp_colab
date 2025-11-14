# vnlp_colab/dep/dep_colab.py
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
Dependency Parser (DP) module for VNLP Colab.

This module provides the high-level DependencyParser API and implementations
for SPUContextDP and the dependency-aware TreeStackDP, all refactored for
Keras 3 and high performance.
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
from vnlp_colab.dep.dep_utils_colab import (
    create_spucontext_dp_model, process_dp_input,
    decode_arc_label_vector, dp_pos_to_displacy_format
)
from vnlp_colab.dep.dep_treestack_utils_colab import (
    create_treestack_dp_model, process_treestack_dp_input
)
from vnlp_colab.stemmer.stemmer_colab import StemmerAnalyzer, get_stemmer_analyzer
from vnlp_colab.pos.pos_colab import PoSTagger

logger = logging.getLogger(__name__)

# --- Model & Resource Configuration ---
_MODEL_CONFIGS = {
    'SPUContextDP': {
        'weights_prod': ("DP_SPUContext_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_SPUContext_prod.weights"),
        'weights_eval': ("DP_SPUContext_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_SPUContext_eval.weights"),
        'word_embedding_matrix': ("SPUTokenized_word_embedding_16k.matrix", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"),
        'spu_tokenizer': "SPU_word_tokenizer_16k.model",
        'label_tokenizer': "DP_label_tokenizer.json",
        'params': {'sentence_max_len': 40, 'word_embedding_dim': 128, 'num_rnn_stacks': 2, 'rnn_units_multiplier': 2, 'fc_units_multiplier': (2, 1), 'dropout': 0.2}
    },
    'TreeStackDP': {
        'weights_prod': ("DP_TreeStack_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_TreeStack_prod.weights"),
        'weights_eval': ("DP_TreeStack_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_TreeStack_eval.weights"),
        'word_embedding_matrix': ("TBWTokenized_word_embedding.matrix", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/TBWTokenized_word_embedding.matrix"),
        'word_tokenizer': "TB_word_tokenizer.json",
        'morph_tag_tokenizer': "Stemmer_morph_tag_tokenizer.json",
        'pos_label_tokenizer': "PoS_label_tokenizer.json",
        'dp_label_tokenizer': "DP_label_tokenizer.json",
        'params': {'word_embedding_vector_size': 128, 'pos_embedding_vector_size': 8, 'num_rnn_stacks': 2, 'tag_num_rnn_units': 128, 'lc_num_rnn_units': 384, 'lc_arc_label_num_rnn_units': 384, 'rc_num_rnn_units': 384, 'fc_units_multipliers': (8, 4), 'word_form': 'whole', 'dropout': 0.2, 'sentence_max_len': 40, 'tag_max_len': 15}
    }
}

# --- Singleton Caching for Model Instances ---
_MODEL_INSTANCE_CACHE: Dict[str, Any] = {}


class SPUContextDP:
    """
    SentencePiece Unigram Context Dependency Parser.
    Optimized with tf.function for high-performance inference.
    """
    def __init__(self, evaluate: bool = False):
        logger.info(f"Initializing SPUContextDP model (evaluate={evaluate})...")
        config = _MODEL_CONFIGS['SPUContextDP']
        self.params_config = config['params'] 
        cache_dir = get_vnlp_cache_dir()

        spu_tokenizer_path = get_resource_path("vnlp_colab.resources", config['spu_tokenizer'])
        label_tokenizer_path = get_resource_path("vnlp_colab.dep.resources", config['label_tokenizer'])
        
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        embedding_path = download_resource(*config['word_embedding_matrix'], cache_dir)

        self.spu_tokenizer_word = spm.SentencePieceProcessor(model_file=str(spu_tokenizer_path))
        self.tokenizer_label = load_keras_tokenizer(label_tokenizer_path)
        self.label_vocab_size = len(self.tokenizer_label.word_index)
        self.arc_label_vector_len = self.params_config['sentence_max_len'] + 1 + self.label_vocab_size + 1
        word_embedding_matrix = np.load(embedding_path)

        # --- MODIFIED: Explicitly pass arguments to prevent TypeErrors ---
        params = self.params_config
        num_rnn_units = params['word_embedding_dim'] * params['rnn_units_multiplier']
        
        self.model = create_spucontext_dp_model(
            vocab_size=self.spu_tokenizer_word.get_piece_size(),
            arc_label_vector_len=self.arc_label_vector_len,
            word_embedding_dim=params['word_embedding_dim'],
            word_embedding_matrix=np.zeros_like(word_embedding_matrix),
            num_rnn_units=num_rnn_units,
            num_rnn_stacks=params['num_rnn_stacks'],
            fc_units_multiplier=params['fc_units_multiplier'],
            dropout=params['dropout']
        )
        with open(weights_path, 'rb') as fp: model_weights = pickle.load(fp)
        self.model.set_weights([word_embedding_matrix] + model_weights)
        self._initialize_compiled_predict_step()
        logger.info("SPUContextDP model initialized successfully.")

    def _initialize_compiled_predict_step(self):
        input_signature = [
            tf.TensorSpec(shape=(1, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(1, 40, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(1, 40, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(1, 40, self.arc_label_vector_len), dtype=tf.float32),
        ]

        @tf.function(input_signature=input_signature)
        def predict_step(word, left_ctx, right_ctx, lc_arc_label_history):
            return self.model([word, left_ctx, right_ctx, lc_arc_label_history], training=False)
        self.compiled_predict_step = predict_step

    def predict(self, tokens: List[str]) -> List[Tuple[int, str, int, str]]:
        if not tokens: return []
        arcs, labels = [], []
        for t in range(len(tokens)):
            inputs_np = process_dp_input(
                t, tokens, self.spu_tokenizer_word, self.tokenizer_label,
                self.arc_label_vector_len, arcs, labels
            )
            inputs_tf = [tf.convert_to_tensor(arr) for arr in inputs_np]
            logits = self.compiled_predict_step(*inputs_tf).numpy()[0]
            arc, label = decode_arc_label_vector(logits, self.params_config['sentence_max_len'], self.label_vocab_size)
            arcs.append(arc)
            labels.append(label)
        
        return [(i + 1, token, arcs[i], self.tokenizer_label.sequences_to_texts([[labels[i]]])[0] or "UNK")
                for i, token in enumerate(tokens)]

class TreeStackDP:
    """Implementation for the TreeStack Dependency Parser."""
    def __init__(self, stemmer_analyzer: StemmerAnalyzer, pos_tagger: PoSTagger, evaluate: bool = False):
        logger.info(f"Initializing TreeStackDP model (evaluate={evaluate})...")
        self.stemmer_analyzer = stemmer_analyzer
        self.pos_tagger = pos_tagger
        config = _MODEL_CONFIGS['TreeStackDP']
        self.params = config['params']
        cache_dir = get_vnlp_cache_dir()

        word_tok_path = get_resource_path("vnlp_colab.resources", config['word_tokenizer'])
        morph_tok_path = get_resource_path("vnlp_colab.stemmer.resources", config['morph_tag_tokenizer'])
        pos_tok_path = get_resource_path("vnlp_colab.pos.resources", config['pos_label_tokenizer'])
        dp_tok_path = get_resource_path("vnlp_colab.dep.resources", config['dp_label_tokenizer'])
        
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        embedding_path = download_resource(*config['word_embedding_matrix'], cache_dir)

        self.tokenizer_word = load_keras_tokenizer(word_tok_path)
        self.tokenizer_morph_tag = load_keras_tokenizer(morph_tok_path)
        self.tokenizer_pos = load_keras_tokenizer(pos_tok_path)
        self.tokenizer_label = load_keras_tokenizer(dp_tok_path)
        
        word_embedding_matrix = np.load(embedding_path)
        tag_embedding_matrix = self.stemmer_analyzer.model.layers[5].weights[0].numpy()
        self.arc_label_vector_len = self.params['sentence_max_len'] + 1 + len(self.tokenizer_label.word_index) + 1
        
        self.model = create_treestack_dp_model(
            word_embedding_vocab_size=len(self.tokenizer_word.word_index) + 1,
            word_embedding_matrix=np.zeros_like(word_embedding_matrix),
            pos_vocab_size=len(self.tokenizer_pos.word_index),
            arc_label_vector_len=self.arc_label_vector_len,
            tag_embedding_matrix=tag_embedding_matrix,
            **self.params
        )
        with open(weights_path, 'rb') as fp: model_weights = pickle.load(fp)
        self.model.set_weights([model_weights[0], word_embedding_matrix] + model_weights[1:])
        logger.info("TreeStackDP model initialized successfully.")

    def predict(self, tokens: List[str]) -> List[Tuple[int, str, int, str]]:
        if not tokens: return []
        
        sentence_analyses = self.stemmer_analyzer.predict(tokens)
        pos_results_tuples = self.pos_tagger.predict(tokens)
        pos_tags = [tag for _, tag in pos_results_tuples]
        
        arcs, labels = [], []
        for t in range(len(tokens)):
            x_inputs = process_treestack_dp_input(
                tokens, sentence_analyses, pos_tags, arcs, labels,
                self.tokenizer_word, self.tokenizer_morph_tag, self.tokenizer_pos, self.tokenizer_label,
                self.params['word_form'], self.params['sentence_max_len'], self.params['tag_max_len'],
                self.arc_label_vector_len
            )
            logits = self.model(x_inputs, training=False).numpy()[0]
            arc, label = decode_arc_label_vector(logits, self.params['sentence_max_len'], len(self.tokenizer_label.word_index))
            arcs.append(arc)
            labels.append(label)

        return [(i + 1, token, arcs[i], self.tokenizer_label.sequences_to_texts([[labels[i]]])[0] or "UNK")
                for i, token in enumerate(tokens)]

class DependencyParser:
    """Main API class for Dependency Parser implementations."""
    def __init__(self, model: str = 'SPUContextDP', evaluate: bool = False):
        self.available_models = ['SPUContextDP', 'TreeStackDP']
        if model not in self.available_models:
            raise ValueError(f"'{model}' is not a valid model. Try one of {self.available_models}")

        cache_key = f"dp_{model}_{'eval' if evaluate else 'prod'}"
        if cache_key not in _MODEL_INSTANCE_CACHE:
            logger.info(f"Instance for '{cache_key}' not found. Creating new one.")
            if model == 'SPUContextDP':
                instance = SPUContextDP(evaluate)
            elif model == 'TreeStackDP':
                stemmer = get_stemmer_analyzer(evaluate)
                pos_tagger = PoSTagger(model='TreeStackPoS', evaluate=evaluate) 
                instance = TreeStackDP(stemmer, pos_tagger, evaluate)
            _MODEL_INSTANCE_CACHE[cache_key] = instance
        else:
            logger.info(f"Found cached instance for '{cache_key}'.")
        self.instance: Union[SPUContextDP, TreeStackDP] = _MODEL_INSTANCE_CACHE[cache_key]

    def predict(
        self, tokens: List[str], displacy_format: bool = False,
        pos_result: List[Tuple[str, str]] = None
    ) -> Union[List[Tuple[int, str, int, str]], List[Dict]]:
        dp_result = self.instance.predict(tokens)
        
        if displacy_format:
            return dp_pos_to_displacy_format(dp_result, pos_result)
        return dp_result