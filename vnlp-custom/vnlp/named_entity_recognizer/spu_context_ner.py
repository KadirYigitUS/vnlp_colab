# File: vnlp/named_entity_recognizer/spu_context_ner.py
# --- FULLY REFACTORED AND OPTIMIZED VERSION ---

from typing import List, Tuple
import pickle
import os
import logging

import numpy as np
import tensorflow as tf
import sentencepiece as spm

from ..tokenizer import TreebankWordTokenize
from ..utils import check_and_download, load_keras_tokenizer
from .utils import ner_to_displacy_format
from ._spu_context_utils import create_spucontext_ner_model, process_single_word_input

# --- Constants and Global Setup ---
RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")
PROD_WEIGHTS_LOC = os.path.join(RESOURCES_PATH, "NER_SPUContext_prod.weights")
EVAL_WEIGHTS_LOC = os.path.join(RESOURCES_PATH, "NER_SPUContext_eval.weights")
WORD_EMBEDDING_MATRIX_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources/SPUTokenized_word_embedding_16k.matrix'))
PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/NER_SPUContext_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/NER_SPUContext_eval.weights"
WORD_EMBEDDING_MATRIX_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"
SPU_TOKENIZER_WORD_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources/SPU_word_tokenizer_16k.model'))
TOKENIZER_LABEL_LOC = os.path.join(RESOURCES_PATH, "NER_label_tokenizer.json")

# --- Model & Data Hyperparameters ---
TOKEN_PIECE_MAX_LEN = 8
SENTENCE_MAX_LEN = 40
spu_tokenizer_word = spm.SentencePieceProcessor(model_file=SPU_TOKENIZER_WORD_LOC)
tokenizer_label = load_keras_tokenizer(TOKENIZER_LABEL_LOC)
label_index_word = tokenizer_label.index_word

# Use consistent, unambiguous vocabulary sizes
WORD_EMBEDDING_VOCAB_SIZE = spu_tokenizer_word.get_piece_size()
NUM_ENTITY_CLASSES = len(tokenizer_label.word_index) + 1  # +1 for padding token 0

WORD_EMBEDDING_VECTOR_SIZE = 128
NUM_RNN_STACKS = 2
RNN_UNITS_MULTIPLIER = 2
NUM_RNN_UNITS = WORD_EMBEDDING_VECTOR_SIZE * RNN_UNITS_MULTIPLIER
FC_UNITS_MULTIPLIER = (2, 1)
DROPOUT = 0.2


class SPUContextNER:
    """
    SentencePiece Unigram Context Named Entity Recognizer class.

    This implementation is optimized with tf.function for high-performance
    inference. It uses an autoregressive mechanism, where the prediction for
    each token is conditioned on the predictions of previous tokens.
    """
    def __init__(self, evaluate: bool):
        """
        Initializes the model, loads weights, and compiles the prediction function.
        This is a heavyweight operation managed by a singleton factory.
        """
        self.spu_tokenizer_word = spu_tokenizer_word
        self.tokenizer_label = tokenizer_label
        self._label_index_word = {i: w for w, i in self.tokenizer_label.word_index.items()}

        # 1. Select and download resources
        model_weights_loc = EVAL_WEIGHTS_LOC if evaluate else PROD_WEIGHTS_LOC
        model_weights_link = EVAL_WEIGHTS_LINK if evaluate else PROD_WEIGHTS_LINK
        check_and_download(WORD_EMBEDDING_MATRIX_LOC, WORD_EMBEDDING_MATRIX_LINK)
        check_and_download(model_weights_loc, model_weights_link)

        # 2. Build the Keras model with a placeholder for embeddings
        self.model = create_spucontext_ner_model(
            TOKEN_PIECE_MAX_LEN, SENTENCE_MAX_LEN, WORD_EMBEDDING_VOCAB_SIZE,
            (NUM_ENTITY_CLASSES - 1), # Pass original number of classes, model adds +1 internally
            WORD_EMBEDDING_VECTOR_SIZE, np.zeros((WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE)),
            NUM_RNN_UNITS, NUM_RNN_STACKS, FC_UNITS_MULTIPLIER, DROPOUT
        )
        
        # 3. Load weights from disk and set them robustly
        word_embedding_matrix = np.load(WORD_EMBEDDING_MATRIX_LOC)
        with open(model_weights_loc, 'rb') as fp:
            model_weights = pickle.load(fp)

        # The non-trainable embedding matrix is expected to be the first weight.
        # The remaining weights from the pickle file are for the trainable layers.
        num_trainable_weights_in_model = len(self.model.trainable_weights)
        num_weights_in_file = len(model_weights)
        assert num_trainable_weights_in_model == num_weights_in_file, \
            f"Weight mismatch: Model expects {num_trainable_weights_in_model} trainable weights, but file has {num_weights_in_file}."
        
        full_weights = [word_embedding_matrix] + model_weights
        self.model.set_weights(full_weights)
        
        # 4. Compile the prediction step into a static graph for speed
        self._initialize_compiled_predict_step()

    def _initialize_compiled_predict_step(self):
        """Creates a compiled TensorFlow function for the model's forward pass."""
        input_signature = [
            tf.TensorSpec(shape=(None, TOKEN_PIECE_MAX_LEN), dtype=tf.int32, name="word_input"),
            tf.TensorSpec(shape=(None, SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), dtype=tf.int32, name="left_context_input"),
            tf.TensorSpec(shape=(None, SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), dtype=tf.int32, name="right_context_input"),
            tf.TensorSpec(shape=(None, SENTENCE_MAX_LEN, NUM_ENTITY_CLASSES), dtype=tf.float32, name="lc_entity_input"),
        ]

        @tf.function(input_signature=input_signature)
        def predict_step(word_input, left_context_input, right_context_input, lc_entity_input):
            return self.model([word_input, left_context_input, right_context_input, lc_entity_input], training=False)
            
        self.compiled_predict_step = predict_step

    def predict(self, sentence: str, displacy_format: bool = False) -> List[Tuple[str, str]]:
        """
        Predicts named entities for a sentence using an optimized autoregressive loop.
        """
        if not sentence or not sentence.strip():
            return []

        tokenized_sentence = TreebankWordTokenize(sentence)
        num_tokens = len(tokenized_sentence)
        int_preds = []
        
        for t in range(num_tokens):
            # 1. Prepare inputs using the optimized helper function
            X_numpy = process_single_word_input(
                t, tokenized_sentence, self.spu_tokenizer_word, self.tokenizer_label, int_preds
            )
            
            # 2. Convert inputs to Tensors and call the compiled prediction function
            X_tensors = [tf.convert_to_tensor(arr) for arr in X_numpy]
            logits = self.compiled_predict_step(*X_tensors).numpy()[0]
            
            # 3. Decode result and update state for the next iteration
            int_pred = np.argmax(logits, axis=-1)
            int_preds.append(int_pred)

        # 4. Convert final integer predictions to text labels using a fast lookup
        preds = [self._label_index_word.get(p, 'O') for p in int_preds]
        ner_result = list(zip(tokenized_sentence, preds))

        return ner_to_displacy_format(sentence, ner_result) if displacy_format else ner_result