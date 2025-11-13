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
from vnlp_colab.utils_colab import download_resource, load_keras_tokenizer, get_vnlp_cache_dir
from vnlp_colab.pos.pos_utils_colab import create_spucontext_pos_model, process_pos_input
from vnlp_colab.pos.pos_treestack_utils_colab import create_treestack_pos_model, process_treestack_pos_input
from vnlp_colab.stemmer.stemmer_colab import StemmerAnalyzer, get_stemmer_analyzer

logger = logging.getLogger(__name__)

# --- Model & Resource Configuration ---
_MODEL_CONFIGS = {
    'SPUContextPoS': {
        'weights_prod': ("PoS_SPUContext_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_SPUContext_prod.weights"),
        'weights_eval': ("PoS_SPUContext_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_SPUContext_eval.weights"),
        'word_embedding_matrix': ("SPUTokenized_word_embedding_16k.matrix", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"),
        'spu_tokenizer': ("SPU_word_tokenizer_16k.model", "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/resources/SPU_word_tokenizer_16k.model"),
        'label_tokenizer': ("PoS_label_tokenizer.json", "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/part_of_speech_tagger/resources/PoS_label_tokenizer.json"),
        'params': { 'word_embedding_dim': 128, 'num_rnn_stacks': 1, 'rnn_units_multiplier': 1, 'fc_units_multiplier': (2, 1), 'dropout': 0.2 }
    },
    'TreeStackPoS': {
        'weights_prod': ("PoS_TreeStack_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_TreeStack_prod.weights"),
        'weights_eval': ("PoS_TreeStack_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_TreeStack_eval.weights"),
        'word_embedding_matrix': ("TBWTokenized_word_embedding.matrix", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/TBWTokenized_word_embedding.matrix"),
        'word_tokenizer': ("TB_word_tokenizer.json", "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/resources/TB_word_tokenizer.json"),
        'morph_tag_tokenizer': ("Stemmer_morph_tag_tokenizer.json", "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/stemmer_morph_analyzer/resources/Stemmer_morph_tag_tokenizer.json"),
        'pos_label_tokenizer': ("PoS_label_tokenizer.json", "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/vnlp/part_of_speech_tagger/resources/PoS_label_tokenizer.json"),
        'params': { 'word_embedding_vector_size': 128, 'num_rnn_stacks': 2, 'tag_num_rnn_units': 128, 'lc_num_rnn_units': 256, 'rc_num_rnn_units': 256, 'fc_units_multipliers': (2, 1), 'word_form': 'whole', 'dropout': 0.2, 'sentence_max_len': 40, 'tag_max_len': 15 }
    }
}

# --- Singleton Caching for Model Instances ---
_MODEL_INSTANCE_CACHE: Dict[str, Any] = {}


class SPUContextPoS:
    """
    SentencePiece Unigram Context Part-of-Speech Tagger.

    Optimized with tf.function for high-performance inference. It uses an
    autoregressive mechanism, where the prediction for each token is conditioned
    on the predictions of previous tokens.
    """
    def __init__(self, evaluate: bool = False):
        """
        Initializes the model, loads weights, and compiles the prediction function.
        This is a heavyweight operation managed by the singleton factory.
        """     
        logger.info(f"Initializing SPUContextPoS model (evaluate={evaluate})...")
        config = _MODEL_CONFIGS['SPUContextPoS']
        cache_dir = get_vnlp_cache_dir()

        # --- Download and Load Resources ---
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        embedding_matrix_path = download_resource(*config['word_embedding_matrix'], cache_dir)
        spu_tokenizer_path = download_resource(*config['spu_tokenizer'], cache_dir)
        label_tokenizer_path = download_resource(*config['label_tokenizer'], cache_dir)

        self.spu_tokenizer_word = spm.SentencePieceProcessor(model_file=str(spu_tokenizer_path))
        self.tokenizer_label = load_keras_tokenizer(label_tokenizer_path)
        self._label_index_word = {i: w for w, i in self.tokenizer_label.word_index.items()}
        word_embedding_matrix = np.load(embedding_matrix_path)

        # --- Build and Load Model ---
        params = config['params']
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
        
        # The non-trainable embedding matrix is the first weight, followed by trainable weights.
        self.model.set_weights([word_embedding_matrix] + model_weights)
        self._initialize_compiled_predict_step()
        logger.info("SPUContextPoS model initialized successfully.")

    def _initialize_compiled_predict_step(self):
        """Creates a compiled TensorFlow function for the model's forward pass."""
        pos_vocab_size = len(self.tokenizer_label.word_index)
        input_signature = [
            tf.TensorSpec(shape=(1, 8), dtype=tf.int32), # TOKEN_PIECE_MAX_LEN = 8
            tf.TensorSpec(shape=(1, 40, 8), dtype=tf.int32), # SENTENCE_MAX_LEN = 40
            tf.TensorSpec(shape=(1, 40, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(1, 40, pos_vocab_size + 1), dtype=tf.float32),
        ]

        @tf.function(input_signature=input_signature)
        def predict_step(word, left_ctx, right_ctx, lc_pos_history):
            return self.model([word, left_ctx, right_ctx, lc_pos_history], training=False)
        self.compiled_predict_step = predict_step

    def predict(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Predicts PoS tags for a list of tokens using an optimized autoregressive loop.

        Args:
            tokens (List[str]): Input sentence tokens.

        Returns:
            List[Tuple[str, str]]: A list of (token, pos_label) tuples.
        """
        if not tokens: return []

        int_preds: List[int] = []

        for t in range(len(tokens)):
            # 1. Prepare inputs using the optimized helper function. 
            inputs_np = process_pos_input(t, tokens, self.spu_tokenizer_word, self.tokenizer_label, int_preds)
            # 2. Convert inputs to Tensors and call the compiled prediction function.
            inputs_tf = [tf.convert_to_tensor(arr) for arr in inputs_np]
            logits = self.compiled_predict_step(*inputs_tf).numpy()[0]
            # 3. Decode result and update state for the next iteration.
            int_pred = np.argmax(logits, axis=-1)
            int_preds.append(int_pred)
        # 4. Convert final integer predictions to text labels using a fast lookup
        pos_labels = [self._label_index_word.get(p, 'UNK') for p in int_preds]
        return list(zip(tokens, pos_labels))

class TreeStackPoS:
    """
    Tree-stack Part of Speech Tagger class.

    - This Part of Speech Tagger is *inspired* by `Tree-stack LSTM in Transition Based Dependency Parsing <https://aclanthology.org/K18-2012/>`_.
    - "Inspire" is emphasized because this implementation uses the approach of using Morphological Tags, Pre-trained word embeddings and POS tags as input for the model, rather than implementing the exact network proposed in the paper.
    - It achieves 0.89 Accuracy and 0.71 F1_macro_score on test sets of Universal Dependencies 2.9.
    - Input data is processed by NLTK.tokenize.TreebankWordTokenizer.
    - For more details about the training procedure, dataset and evaluation metrics, see `ReadMe <https://github.com/vngrs-ai/VNLP/blob/main/vnlp/part_of_speech_tagger/ReadMe.md>`_.
    """
    def __init__(self, stemmer_analyzer: StemmerAnalyzer, evaluate: bool = False):
        logger.info(f"Initializing TreeStackPoS model (evaluate={evaluate})...")
        self.stemmer_analyzer = stemmer_analyzer
        config = _MODEL_CONFIGS['TreeStackPoS']
        self.params = config['params']
        cache_dir = get_vnlp_cache_dir()
        # Check and download word embedding matrix and model weights
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        embedding_path = download_resource(*config['word_embedding_matrix'], cache_dir)
        self.tokenizer_word = load_keras_tokenizer(download_resource(*config['word_tokenizer'], cache_dir))
        self.tokenizer_morph_tag = load_keras_tokenizer(download_resource(*config['morph_tag_tokenizer'], cache_dir))
        self.tokenizer_pos_label = load_keras_tokenizer(download_resource(*config['pos_label_tokenizer'], cache_dir))
        self._pos_index_word = {i: w for w, i in self.tokenizer_pos_label.word_index.items()}
        # Load Word embedding matrix
        word_embedding_matrix = np.load(embedding_path)
        tag_embedding_matrix = self.stemmer_analyzer.model.layers[5].weights[0].numpy()
        # Load Model weights
        self.model = create_treestack_pos_model(
            word_embedding_vocab_size=len(self.tokenizer_word.word_index) + 1,
            pos_vocab_size=len(self.tokenizer_pos_label.word_index),
            tag_embedding_matrix=tag_embedding_matrix,
            **self.params
        )
        with open(weights_path, 'rb') as fp:
            model_weights = pickle.load(fp)
        
        self.model.set_weights([model_weights[0], word_embedding_matrix] + model_weights[1:])
        logger.info("TreeStackPoS model initialized successfully.")

    def predict(self, tokens: List[str]) -> List[Tuple[str, str]]:
                """
        Args:
            sentence:
                Input text(sentence).

        Returns:
             List of (token, pos_label).
        """
        if not tokens: return []
        sentence_analyses = self.stemmer_analyzer.predict(tokens)
        
        pos_int_labels: List[int] = []
        for t in range(len(tokens)):
            x_inputs = process_treestack_pos_input(
                tokens, sentence_analyses, pos_int_labels, self.tokenizer_word,
                self.tokenizer_morph_tag, self.tokenizer_pos_label, self.params['word_form'],
                self.params['sentence_max_len'], self.params['tag_max_len']
            )
            raw_pred = self.model(x_inputs, training=False).numpy()[0]
            pos_int_label = np.argmax(raw_pred, axis=-1)
            pos_int_labels.append(pos_int_label)

        # Converting integer labels to text form 
        pos_labels = [self._pos_index_word.get(p, 'UNK') for p in pos_int_labels]
        return list(zip(tokens, pos_labels))


class PoSTagger:
    """
    Main API class for Part-of-Speech Tagger implementations.

    This class uses a singleton factory to ensure that heavy models are loaded
    into memory only once, making subsequent initializations instantaneous.

    Available models: ['SPUContextPoS']

    Example::
        from pos_colab import PoSTagger
        # First initialization is slow as it downloads and loads the model.
        pos_tagger = PoSTagger(model='SPUContextPoS')
        # Second initialization is instantaneous.
        pos_tagger2 = PoSTagger(model='SPUContextPoS')

        sentence = "Vapurla Beşiktaş'a geçip yürüyerek Maçka Parkı'na ulaştım."
        predictions = pos_tagger.predict(sentence)
        print(predictions)
        # Output: [('Vapurla', 'Noun'), ("Beşiktaş'a", 'Propn'), ('geçip', 'Verb'), ...]
    """
    def __init__(self, model: str = 'SPUContextPoS', evaluate: bool = False):
        self.available_models = ['SPUContextPoS', 'TreeStackPoS']
        if model not in self.available_models:
            raise ValueError(f"'{model}' is not a valid model. Try one of {self.available_models}")

        # Singleton factory logic
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
        """Predicts PoS tags for a pre-tokenized list of words."""
        return self.instance.predict(tokens)


# --- Main Entry Point for Standalone Use ---
def main():
    """Demonstrates and tests the PoS Tagger module."""
    setup_logging()
    logger.info("--- VNLP Colab PoS Tagger Test Suite ---")
    sentence = "Vapurla Beşiktaş'a geçip yürüyerek Maçka Parkı'na ulaştım."
    tokens = TreebankWordTokenize(sentence)

    try:
        logger.info("\n1. Testing SPUContextPoS...")
        pos_spu = PoSTagger(model='SPUContextPoS')
        result_spu = pos_spu.predict(tokens)
        logger.info(f"   Input: {tokens}")
        logger.info(f"   SPUContextPoS Output: {result_spu}")
        assert len(result_spu) == len(tokens) and result_spu[1][1] == 'PROPN'
        logger.info("   SPUContextPoS test PASSED.")
    except Exception as e:
        logger.error(f"   SPUContextPoS test FAILED: {e}", exc_info=True)

    try:
        logger.info("\n2. Testing TreeStackPoS...")
        pos_tree = PoSTagger(model='TreeStackPoS')
        result_tree = pos_tree.predict(tokens)
        logger.info(f"   Input: {tokens}")
        logger.info(f"   TreeStackPoS Output: {result_tree}")
        assert len(result_tree) == len(tokens) and result_tree[1][1] == 'PROPN'
        logger.info("   TreeStackPoS test PASSED.")
    except Exception as e:
        logger.error(f"   TreeStackPoS test FAILED: {e}", exc_info=True)
    
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