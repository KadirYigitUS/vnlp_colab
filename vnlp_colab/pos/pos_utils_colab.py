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
Keras 3 compliant utilities for the SPUContext Part-of-Speech (PoS) Tagger.

This module provides the modernized model creation function for the PoS tagger,
ensuring compatibility with the latest TensorFlow/Keras versions and Colab.
"""
import logging
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Updated import for package structure
from vnlp_colab.utils_colab import create_rnn_stacks, process_word_context

logger = logging.getLogger(__name__)

# --- Model Hyperparameters (as constants for clarity) ---
TOKEN_PIECE_MAX_LEN: int = 8
SENTENCE_MAX_LEN: int = 40

def create_spucontext_pos_model(
    vocab_size: int,
    pos_vocab_size: int,
    word_embedding_dim: int,
    word_embedding_matrix: np.ndarray,
    num_rnn_units: int,
    num_rnn_stacks: int,
    fc_units_multiplier: tuple[int, int],
    dropout: float
) -> keras.Model:
    """
    Builds the SPUContext PoS tagging model using the Keras 3 Functional API.

    This architecture is a 1:1 replica of the original model blueprint, ensuring
    weight compatibility while adhering to modern Keras standards.

    Args:
        vocab_size (int): Vocabulary size for the word embedding layer.
        pos_vocab_size (int): Vocabulary size for the PoS tags. The output layer
            will have `pos_vocab_size + 1` units for the padding token.
        word_embedding_dim (int): Dimension of the word embeddings.
        word_embedding_matrix (np.ndarray): Pre-trained word embedding matrix.
        num_rnn_units (int): Number of units in the GRU layers.
        num_rnn_stacks (int): Number of layers in each RNN stack.
        fc_units_multiplier (tuple[int, int]): Multipliers for the dense layers.
        dropout (float): Dropout rate.

    Returns:
        keras.Model: The compiled Keras model.
    """
    logger.info("Creating Keras 3 SPUContext PoS Tagger model...")

    # --- 1. Define Functional API Inputs ---
    word_input = keras.Input(
        shape=(TOKEN_PIECE_MAX_LEN,), name='word_input', dtype='int32'
    )
    left_context_input = keras.Input(
        shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), name='left_input', dtype='int32'
    )
    right_context_input = keras.Input(
        shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), name='right_input', dtype='int32'
    )
    # Input for previously predicted PoS tags (one-hot encoded)
    lc_pos_input = keras.Input(
        shape=(SENTENCE_MAX_LEN, pos_vocab_size + 1), name='lc_pos_input', dtype='float32'
    )

    # --- 2. Define Reusable Sub-Models ---

    # Shared model for processing token pieces of a single word
    word_embedding_layer = keras.layers.Embedding(
        vocab_size,
        word_embedding_dim,
        embeddings_initializer=keras.initializers.Constant(word_embedding_matrix),
        trainable=False,
        name='WORD_EMBEDDING'
    )
    word_rnn_stack = create_rnn_stacks(num_rnn_stacks, num_rnn_units, dropout)
    word_rnn_model = keras.Sequential(
        [word_embedding_layer, word_rnn_stack], name="WORD_RNN"
    )

    # --- 3. Build the Four Parallel Branches of the Main Graph ---

    # Branch 1: Current word processing
    word_output = word_rnn_model(word_input)

    # Branch 2: Left context processing
    left_context_vectors = keras.layers.TimeDistributed(word_rnn_model)(left_context_input)
    left_context_stack = create_rnn_stacks(num_rnn_stacks, num_rnn_units, dropout)
    left_context_output = left_context_stack(left_context_vectors)

    # Branch 3: Right context processing (processes sequence in reverse)
    right_context_vectors = keras.layers.TimeDistributed(word_rnn_model)(right_context_input)
    right_context_stack = create_rnn_stacks(num_rnn_stacks, num_rnn_units, dropout, go_backwards=True)
    right_context_output = right_context_stack(right_context_vectors)

    # Branch 4: Previously predicted (left) PoS tags processing
    lc_pos_stack = create_rnn_stacks(num_rnn_stacks, num_rnn_units, dropout)
    lc_pos_output = lc_pos_stack(lc_pos_input)

    # --- 4. Concatenate Branches and Add Final FC Layers ---
    concatenated = keras.layers.Concatenate()(
        [word_output, left_context_output, right_context_output, lc_pos_output]
    )

    x = keras.layers.Dense(num_rnn_units * fc_units_multiplier[0], activation='relu')(concatenated)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(num_rnn_units * fc_units_multiplier[1], activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    pos_output = keras.layers.Dense(pos_vocab_size + 1, activation='softmax')(x)

    # --- 5. Build and Return the Final Model ---
    pos_model = keras.Model(
        inputs=[word_input, left_context_input, right_context_input, lc_pos_input],
        outputs=pos_output,
        name='SPUContext_PoS_Model'
    )
    logger.info("SPUContext PoS Tagger model created successfully.")
    return pos_model


def process_pos_input(
    word_index: int,
    sentence_tokens: List[str],
    spu_tokenizer_word: 'spm.SentencePieceProcessor',
    pos_label_tokenizer: tf.keras.preprocessing.text.Tokenizer,
    previous_predictions: List[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares input arrays for a single word for the PoS tagger model.

    This function vectorizes the word and its context, as well as the history of
    previously predicted PoS tags for the autoregressive loop.

    Args:
        word_index (int): Index of the current word in the sentence.
        sentence_tokens (List[str]): List of all tokens in the sentence.
        spu_tokenizer_word: The SentencePiece tokenizer.
        pos_label_tokenizer: The Keras tokenizer for PoS labels.
        previous_predictions (List[int]): A list of integer PoS tag predictions
            for the preceding words in the sentence.

    Returns:
        A tuple of NumPy arrays ready for model input:
        (current_word, left_context, right_context, left_pos_history)
    """
    pos_vocab_size = len(pos_label_tokenizer.word_index) + 1

    # 1. Process word and its context using the utility function
    current_word, left_context, right_context = process_word_context(
        word_index,
        sentence_tokens,
        spu_tokenizer_word,
        SENTENCE_MAX_LEN,
        TOKEN_PIECE_MAX_LEN
    )

    # 2. Create the history of previous PoS tag predictions
    left_pos_history = np.zeros((SENTENCE_MAX_LEN, pos_vocab_size), dtype=np.float32)

    if word_index > 0:
        # Create one-hot vectors for all previous predictions
        one_hot_preds = tf.keras.utils.to_categorical(
            previous_predictions, num_classes=pos_vocab_size
        )
        # Place them in the correct time-steps of the input array
        # The history starts from the `SENTENCE_MAX_LEN - word_index`-th position
        start_pos = SENTENCE_MAX_LEN - word_index
        end_pos = start_pos + len(one_hot_preds)
        left_pos_history[start_pos:end_pos] = one_hot_preds

    # 3. Expand dims to create a batch of 1 for model prediction
    return (
        np.expand_dims(current_word, axis=0),
        np.expand_dims(left_context, axis=0),
        np.expand_dims(right_context, axis=0),
        np.expand_dims(left_pos_history, axis=0)
    )