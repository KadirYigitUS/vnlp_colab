# vnlp_colab/ner/ner_colab.py
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
Named Entity Recognition (NER) module for VNLP Colab.

This module contains the high-level NamedEntityRecognizer API and the underlying
model implementations (SPUContextNER, CharNER), refactored for Keras 3,
performance, and a modern developer experience in Colab.
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
from vnlp_colab.ner.ner_utils_colab import (
    create_spucontext_ner_model, process_ner_input,
    create_charner_model, ner_to_displacy_format
)
from vnlp_colab.tokenizer_colab import WordPunctTokenize


logger = logging.getLogger(__name__)

# --- Model & Resource Configuration ---
_MODEL_CONFIGS = {
    'SPUContextNER': {
        'weights_prod': ("NER_SPUContext_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/NER_SPUContext_prod.weights"),
        'weights_eval': ("NER_SPUContext_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/NER_SPUContext_eval.weights"),
        'word_embedding_matrix': ("SPUTokenized_word_embedding_16k.matrix", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"),
        'spu_tokenizer': "SPU_word_tokenizer_16k.model",
        'label_tokenizer': "NER_label_tokenizer.json",
        'params': {
            'word_embedding_dim': 128, 'num_rnn_stacks': 2, 'rnn_units_multiplier': 2,
            'fc_units_multiplier': (2, 1), 'dropout': 0.2,
        }
    },
    'CharNER': {
        'weights_prod': ("NER_CharNER_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/NER_CharNER_prod.weights"),
        'weights_eval': ("NER_CharNER_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/NER_CharNER_eval.weights"),
        'char_tokenizer': "CharNER_char_tokenizer.json",
        'label_tokenizer': "NER_label_tokenizer.json",
        'params': {
            'char_vocab_size': 150, 'seq_len_max': 256, 'embed_size': 32, 'rnn_dim': 128,
            'num_rnn_stacks': 5, 'mlp_dim': 32, 'num_classes': 5, 'dropout': 0.3,
            'padding_strat': 'post',
        }
    }
}

# --- Singleton Caching for Model Instances ---
_MODEL_INSTANCE_CACHE: Dict[str, Any] = {}


class SPUContextNER:
    """
    SentencePiece Unigram Context Named Entity Recognizer.
    Optimized with tf.function for high-performance inference.
    """
    def __init__(self, evaluate: bool = False):
        logger.info(f"Initializing SPUContextNER model (evaluate={evaluate})...")
        config = _MODEL_CONFIGS['SPUContextNER']
        params = config['params']
        cache_dir = get_vnlp_cache_dir()

        spu_tokenizer_path = get_resource_path("vnlp_colab.resources", config['spu_tokenizer'])
        label_tokenizer_path = get_resource_path("vnlp_colab.ner.resources", config['label_tokenizer'])
        
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        embedding_path = download_resource(*config['word_embedding_matrix'], cache_dir)

        self.spu_tokenizer_word = spm.SentencePieceProcessor(model_file=str(spu_tokenizer_path))
        self.tokenizer_label = load_keras_tokenizer(label_tokenizer_path)
        self._label_index_word = {i: w for w, i in self.tokenizer_label.word_index.items()}
        
        word_embedding_matrix = np.load(embedding_path)
        
        # --- MODIFIED: Explicitly pass arguments to prevent TypeErrors ---
        num_rnn_units = params['word_embedding_dim'] * params['rnn_units_multiplier']
        
        self.model = create_spucontext_ner_model(
            vocab_size=self.spu_tokenizer_word.get_piece_size(),
            entity_vocab_size=len(self.tokenizer_label.word_index),
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
        logger.info("SPUContextNER model initialized successfully.")

    def _initialize_compiled_predict_step(self):
        entity_vocab_size = len(self.tokenizer_label.word_index)
        input_signature = [
            tf.TensorSpec(shape=(1, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(1, 40, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(1, 40, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(1, 40, entity_vocab_size + 1), dtype=tf.float32),
        ]

        @tf.function(input_signature=input_signature)
        def predict_step(word, left_ctx, right_ctx, lc_entity_history):
            return self.model([word, left_ctx, right_ctx, lc_entity_history], training=False)
            
        self.compiled_predict_step = predict_step

    def predict(self, sentence: str, tokens: List[str], displacy_format: bool = False) -> Union[List[Tuple[str, str]], Dict]:
        if not tokens: return []
        num_tokens = len(tokens)
        int_preds: List[int] = []

        for t in range(num_tokens):
            inputs_np = process_ner_input(t, tokens, self.spu_tokenizer_word, self.tokenizer_label, int_preds)
            inputs_tf = [tf.convert_to_tensor(arr) for arr in inputs_np]
            logits = self.compiled_predict_step(*inputs_tf).numpy()[0]
            int_pred = np.argmax(logits, axis=-1)
            int_preds.append(int_pred)

        preds = [self._label_index_word.get(p, 'O') for p in int_preds]
        ner_result = list(zip(tokens, preds))

        return ner_to_displacy_format(sentence, ner_result) if displacy_format else ner_result

class CharNER:
    """Character-Level Named Entity Recognizer."""
    def __init__(self, evaluate: bool = False):
        logger.info(f"Initializing CharNER model (evaluate={evaluate})...")
        config = _MODEL_CONFIGS['CharNER']
        cache_dir = get_vnlp_cache_dir()

        resource_pkg_path = "vnlp_colab.ner.resources"
        char_tokenizer_path = get_resource_path(resource_pkg_path, config['char_tokenizer'])
        label_tokenizer_path = get_resource_path(resource_pkg_path, config['label_tokenizer'])
        
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        
        self.tokenizer_char = load_keras_tokenizer(char_tokenizer_path)
        self.tokenizer_label = load_keras_tokenizer(label_tokenizer_path)
        
        params = config['params']
        self.model = create_charner_model(
            params['char_vocab_size'], params['embed_size'], params['seq_len_max'],
            params['num_rnn_stacks'], params['rnn_dim'], params['mlp_dim'],
            params['num_classes'], params['dropout']
        )
        
        with open(weights_path, 'rb') as fp:
            self.model.set_weights(pickle.load(fp))

        self.seq_len_max = params['seq_len_max']
        self.padding_strat = params['padding_strat']
        logger.info("CharNER model initialized successfully.")

    def _predict_char_level(self, tokens: List[str]) -> np.ndarray:
        text = " ".join(tokens)
        char_sequence = list(text)
        sequences = self.tokenizer_char.texts_to_sequences([char_sequence])
        padded = keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.seq_len_max, padding=self.padding_strat
        )
        raw_pred = self.model(padded, training=False)
        return np.argmax(raw_pred, axis=2).flatten()

    def _charner_decoder(self, tokens: List[str], preds: np.ndarray) -> List[str]:
        lens = [0] + [len(token) + 1 for token in tokens]
        cumsum_of_lens = np.cumsum(lens)
        decoded_entities = []
        for i in range(len(cumsum_of_lens) - 1):
            island = preds[cumsum_of_lens[i] : cumsum_of_lens[i+1] - 1]
            mode_value = 0
            if island.size > 0:
                vals, counts = np.unique(island, return_counts=True)
                mode_value = vals[np.argmax(counts)]
            detokenized = self.tokenizer_label.sequences_to_texts([[mode_value]])[0]
            decoded_entities.append(detokenized or 'O')
        return decoded_entities

    def predict(self, sentence: str, tokens: List[str], displacy_format: bool = False) -> Union[List[Tuple[str, str]], Dict]:
        internal_tokens = WordPunctTokenize(sentence)
        if len(" ".join(internal_tokens)) > self.seq_len_max:
            mid = len(internal_tokens) // 2
            first_half_sent = " ".join(internal_tokens[:mid])
            second_half_sent = " ".join(internal_tokens[mid:])
            first_half_res = self.predict(first_half_sent, [], displacy_format)
            second_half_res = self.predict(second_half_sent, [], displacy_format)
            if displacy_format:
                second_half_res['text'] = first_half_res['text'] + " " + second_half_res['text']
                second_half_res['ents'].extend(first_half_res['ents'])
                return second_half_res
            else:
                return first_half_res + second_half_res


        char_preds = self._predict_char_level(internal_tokens)
        decoded_entities = self._charner_decoder(internal_tokens, char_preds)
        ner_result = list(zip(internal_tokens, decoded_entities))
        
        return ner_to_displacy_format(sentence, ner_result) if displacy_format else ner_result


class NamedEntityRecognizer:
    """Main API class for Named Entity Recognizer implementations."""
    def __init__(self, model: str = 'SPUContextNER', evaluate: bool = False):
        self.available_models = ['SPUContextNER', 'CharNER']
        if model not in self.available_models:
            raise ValueError(f"'{model}' is not a valid model. Try one of {self.available_models}")
        self.model_name = model

        cache_key = f"ner_{model}_{'eval' if evaluate else 'prod'}"
        if cache_key not in _MODEL_INSTANCE_CACHE:
            logger.info(f"Instance for '{cache_key}' not found in cache. Creating new one.")
            if model == 'SPUContextNER':
                _MODEL_INSTANCE_CACHE[cache_key] = SPUContextNER(evaluate)
            elif model == 'CharNER':
                _MODEL_INSTANCE_CACHE[cache_key] = CharNER(evaluate)
        else:
            logger.info(f"Found cached instance for '{cache_key}'.")
        self.instance: Union[SPUContextNER, CharNER] = _MODEL_INSTANCE_CACHE[cache_key]

    def predict(self, sentence: str, tokens: List[str], displacy_format: bool = False) -> Union[List[Tuple[str, str]], Dict]:
        return self.instance.predict(sentence, tokens, displacy_format)


def main():
    """Demonstrates and tests the NER module functions."""
    from vnlp_colab.utils_colab import setup_logging
    setup_logging()
    logger.info("--- VNLP Colab NER Test Suite ---")
    
    try:
        logger.info("\n1. Testing SPUContextNER...")
        ner_spu = NamedEntityRecognizer(model='SPUContextNER')
        sentence1 = "Benim adım Melikşah, İstanbul'da yaşıyorum."
        tokens1 = WordPunctTokenize(sentence1)
        result1 = ner_spu.predict(sentence1, tokens1)
        logger.info(f"   Input: '{sentence1}'")
        logger.info(f"   Output: {result1}")
        assert len(result1) > 0 and result1[2][1] == 'PER'
        logger.info("   SPUContextNER test PASSED.")
    except Exception as e:
        logger.error(f"   SPUContextNER test FAILED: {e}", exc_info=True)

    try:
        logger.info("\n2. Testing CharNER...")
        ner_char = NamedEntityRecognizer(model='CharNER')
        sentence2 = "VNGRS AI Takımı'nda çalışıyorum."
        result2 = ner_char.predict(sentence2, []) # Tokens ignored by CharNER
        logger.info(f"   Input: '{sentence2}'")
        logger.info(f"   Output: {result2}")
        assert len(result2) > 0 and result2[0][1] == 'ORG'
        logger.info("   CharNER test PASSED.")
    except Exception as e:
        logger.error(f"   CharNER test FAILED: {e}", exc_info=True)
        
    logger.info("\n3. Testing Singleton Caching...")
    import time
    start_time = time.time()
    _ = NamedEntityRecognizer(model='SPUContextNER')
    end_time = time.time()
    logger.info(f"   Re-initialization took: {end_time - start_time:.4f} seconds.")
    assert (end_time - start_time) < 0.1, "Caching failed, re-initialization was too slow."
    logger.info("   Singleton Caching test PASSED.")


if __name__ == "__main__":
    main()