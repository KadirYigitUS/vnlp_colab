# File: /home/ben/miniconda3/envs/bookanalysis/lib/python3.12/site-packages/vnlp/dependency_parser/spu_context_dp.py

from typing import List, Tuple
import pickle
import os

import numpy as np
import tensorflow as tf
import sentencepiece as spm

from ..tokenizer import TreebankWordTokenize
from ..utils import check_and_download, load_keras_tokenizer, process_word_context
from .utils import dp_pos_to_displacy_format, decode_arc_label_vector
from ._spu_context_utils import create_spucontext_dp_model

# --- Constants and Global Setup ---
RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")
PROD_WEIGHTS_LOC = os.path.join(RESOURCES_PATH, "DP_SPUContext_prod.weights")
EVAL_WEIGHTS_LOC = os.path.join(RESOURCES_PATH, "DP_SPUContext_eval.weights")
WORD_EMBEDDING_MATRIX_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources/SPUTokenized_word_embedding_16k.matrix'))
SPU_TOKENIZER_WORD_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources/SPU_word_tokenizer_16k.model'))
TOKENIZER_LABEL_LOC = os.path.join(RESOURCES_PATH, "DP_label_tokenizer.json")
PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_SPUContext_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_SPUContext_eval.weights"
WORD_EMBEDDING_MATRIX_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"

# --- Model Hyperparameters ---
TOKEN_PIECE_MAX_LEN = 8
SENTENCE_MAX_LEN = 40
WORD_EMBEDDING_VECTOR_SIZE = 128
NUM_RNN_STACKS = 2
RNN_UNITS_MULTIPLIER = 2
NUM_RNN_UNITS = WORD_EMBEDDING_VECTOR_SIZE * RNN_UNITS_MULTIPLIER
FC_UNITS_MULTIPLIER = (2, 1)
DROPOUT = 0.2

class SPUContextDP:
    """Final Optimized and Corrected Dependency Parser."""
    def __init__(self, evaluate: bool):
        self.spu_tokenizer_word = spm.SentencePieceProcessor(model_file=SPU_TOKENIZER_WORD_LOC)
        self.tokenizer_label = load_keras_tokenizer(TOKENIZER_LABEL_LOC)
        self.LABEL_VOCAB_SIZE = len(self.tokenizer_label.word_index)
        self.WORD_EMBEDDING_VOCAB_SIZE = self.spu_tokenizer_word.get_piece_size()
        self.ARC_LABEL_VECTOR_LEN = SENTENCE_MAX_LEN + 1 + self.LABEL_VOCAB_SIZE + 1

        check_and_download(WORD_EMBEDDING_MATRIX_LOC, WORD_EMBEDDING_MATRIX_LINK)
        MODEL_WEIGHTS_LOC = EVAL_WEIGHTS_LOC if evaluate else PROD_WEIGHTS_LOC
        MODEL_WEIGHTS_LINK = EVAL_WEIGHTS_LINK if evaluate else PROD_WEIGHTS_LINK
        check_and_download(MODEL_WEIGHTS_LOC, MODEL_WEIGHTS_LINK)

        word_embedding_matrix = np.load(WORD_EMBEDDING_MATRIX_LOC)
        with open(MODEL_WEIGHTS_LOC, 'rb') as fp:
            model_weights = pickle.load(fp)
        model_weights.insert(0, word_embedding_matrix)

        self.model = create_spucontext_dp_model(
            TOKEN_PIECE_MAX_LEN, SENTENCE_MAX_LEN, self.WORD_EMBEDDING_VOCAB_SIZE,
            self.ARC_LABEL_VECTOR_LEN, WORD_EMBEDDING_VECTOR_SIZE, word_embedding_matrix,
            NUM_RNN_UNITS, NUM_RNN_STACKS, FC_UNITS_MULTIPLIER, DROPOUT
        )
        
        self.model.set_weights(model_weights)
        self._initialize_predict_step()

    def _vectorize_arc_label(self, w_idx, arcs, labels):
        """Internal helper to create an arc-label vector from previous predictions."""
        arc = arcs[w_idx]
        label = labels[w_idx]
        arc_vector = tf.keras.utils.to_categorical(arc, num_classes=SENTENCE_MAX_LEN + 1)
        label_vector = tf.keras.utils.to_categorical(label, num_classes=self.LABEL_VOCAB_SIZE + 1)
        return np.concatenate([arc_vector, label_vector])

    def _initialize_predict_step(self):
        """Compiles a simple, single-step model call for use in a loop."""
        @tf.function(input_signature=[
            tf.TensorSpec(shape=(1, TOKEN_PIECE_MAX_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(1, SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(1, SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(1, SENTENCE_MAX_LEN, self.ARC_LABEL_VECTOR_LEN), dtype=tf.float32),
        ])
        def predict_step(word, left_ctx, right_ctx, lc_arc_label):
            return self.model([word, left_ctx, right_ctx, lc_arc_label], training=False)
        self.predict_step = predict_step

    def predict(self, sentence: str, displacy_format: bool = False, pos_result: List[Tuple[str, str]] = None) -> List[Tuple[int, str, int, str]]:
        tokenized_sentence = TreebankWordTokenize(sentence)
        num_tokens = len(tokenized_sentence)

        if num_tokens > SENTENCE_MAX_LEN:
            raise ValueError(f'Sentence is too long ({num_tokens} tokens). Max is {SENTENCE_MAX_LEN}.')

        # --- 1. Full Pre-computation Step ---
        # Pre-calculate and convert to Tensors ONCE, outside the loop.
        all_words_np, all_left_np, all_right_np = [], [], []
        for t in range(num_tokens):
            word, left, right = process_word_context(
                t, tokenized_sentence, self.spu_tokenizer_word, SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN
            )
            all_words_np.append(word)
            all_left_np.append(left)
            all_right_np.append(right)
        
        all_words_tf = tf.constant(np.array(all_words_np), dtype=tf.int32)
        all_left_tf = tf.constant(np.array(all_left_np), dtype=tf.int32)
        all_right_tf = tf.constant(np.array(all_right_np), dtype=tf.int32)

        # --- 2. Optimized Prediction Loop ---
        arcs, labels = [], []
        lc_arc_label_input_np = np.zeros((1, SENTENCE_MAX_LEN, self.ARC_LABEL_VECTOR_LEN), dtype=np.float32)

        for t in range(num_tokens):
            # a. Quickly build the dependent input using fast NumPy
            if t > 0:
                # This is much faster than rebuilding the whole array every time
                new_vector = self._vectorize_arc_label(t - 1, arcs, labels)
                position = (SENTENCE_MAX_LEN - t) + (t - 1)
                lc_arc_label_input_np[0, position] = new_vector
            
            # b. Slice the pre-computed Tensors (very fast)
            word_input = all_words_tf[t:t+1]
            left_input = all_left_tf[t:t+1]
            right_input = all_right_tf[t:t+1]
            
            # c. Call the compiled function
            logits = self.predict_step(
                word_input, left_input, right_input, tf.constant(lc_arc_label_input_np)
            ).numpy()[0]
            
            # d. Decode and store results
            arc, label = decode_arc_label_vector(logits, SENTENCE_MAX_LEN, self.LABEL_VOCAB_SIZE)
            arcs.append(arc)
            labels.append(label)

        # --- 3. Format Final Result ---
        dp_result = [
            (idx + 1, word, int(arcs[idx]), self.tokenizer_label.sequences_to_texts([[labels[idx]]])[0])
            for idx, word in enumerate(tokenized_sentence)
        ]

        return dp_pos_to_displacy_format(dp_result, pos_result) if displacy_format else dp_result