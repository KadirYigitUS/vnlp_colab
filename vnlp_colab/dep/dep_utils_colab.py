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
Keras 3 compliant utilities for the SPUContext Dependency Parser (DP).

This module provides the modernized model creation function for the DP model,
ensuring compatibility with the latest TensorFlow/Keras versions and Colab.
"""
import logging
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Updated import for package structure
from vnlp_colab.utils_colab import create_rnn_stacks, process_word_context

logger = logging.getLogger(__name__)

# --- Model Hyperparameters (as constants for clarity) ---
TOKEN_PIECE_MAX_LEN: int = 8
SENTENCE_MAX_LEN: int = 40

# --- SPUContextDP Utilities ---

def create_spucontext_dp_model(
    vocab_size: int,
    arc_label_vector_len: int,
    word_embedding_dim: int,
    word_embedding_matrix: np.ndarray,
    num_rnn_units: int,
    num_rnn_stacks: int,
    fc_units_multiplier: tuple[int, int],
    dropout: float
) -> keras.Model:
       """
    Builds the SPUContext Dependency Parser model using the Keras 3 Functional API.

    This architecture is a 1:1 replica of the original model blueprint, ensuring
    weight compatibility while adhering to modern Keras standards.

    Args:
        vocab_size (int): Vocabulary size for the word embedding layer.
        arc_label_vector_len (int): The total length of the concatenated one-hot
            vectors for arc and label predictions.
        word_embedding_dim (int): Dimension of the word embeddings.
        word_embedding_matrix (np.ndarray): Pre-trained word embedding matrix.
        num_rnn_units (int): Number of units in the GRU layers.
        num_rnn_stacks (int): Number of layers in each RNN stack.
        fc_units_multiplier (tuple[int, int]): Multipliers for the dense layers.
        dropout (float): Dropout rate.

    Returns:
        keras.Model: The compiled Keras model.
    """
    logger.info("Creating Keras 3 SPUContext Dependency Parser model...")

    # --- 1. Define Functional API Inputs ---
    word_input = keras.Input(shape=(TOKEN_PIECE_MAX_LEN,), name='word_input', dtype='int32')
    left_context_input = keras.Input(shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), name='left_context_input', dtype='int32')
    right_context_input = keras.Input(shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), name='right_context_input', dtype='int32')
    lc_arc_label_input = keras.Input(shape=(SENTENCE_MAX_LEN, arc_label_vector_len), name='lc_arc_label_input', dtype='float32')

    # --- 2. Define Reusable Sub-Models ---
    word_embedding_layer = keras.layers.Embedding(
        vocab_size, word_embedding_dim,
        embeddings_initializer=keras.initializers.Constant(word_embedding_matrix),
        trainable=False, name='WORD_EMBEDDING'
    )
    word_rnn_stack = create_rnn_stacks(num_rnn_stacks, num_rnn_units, dropout)
    word_rnn_model = keras.Sequential([word_embedding_layer, word_rnn_stack], name="WORD_RNN")

    # --- 3. Build the Four Parallel Branches of the Main Graph ---
    # Branch 1: Current word
    word_output = word_rnn_model(word_input)

    # Branch 2: Left context
    left_context_vectors = keras.layers.TimeDistributed(word_rnn_model)(left_context_input)
    left_context_stack = create_rnn_stacks(num_rnn_stacks, num_rnn_units, dropout)
    left_context_output = left_context_stack(left_context_vectors)

    # Branch 3: Right context
    right_context_vectors = keras.layers.TimeDistributed(word_rnn_model)(right_context_input)
    right_context_stack = create_rnn_stacks(num_rnn_stacks, num_rnn_units, dropout, go_backwards=True)
    right_context_output = right_context_stack(right_context_vectors)

    # Branch 4: Previously predicted (left) arc-labels
    lc_arc_label_stack = create_rnn_stacks(num_rnn_stacks, num_rnn_units, dropout)
    lc_arc_label_output = lc_arc_label_stack(lc_arc_label_input)

    # --- 4. Concatenate Branches and Add Final FC Layers ---
    concatenated = keras.layers.Concatenate()([word_output, left_context_output, right_context_output, lc_arc_label_output])

    x = keras.layers.Dense(num_rnn_units * fc_units_multiplier[0], activation='relu')(concatenated)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(num_rnn_units * fc_units_multiplier[1], activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    arc_label_output = keras.layers.Dense(arc_label_vector_len, activation='sigmoid')(x)

    # --- 5. Build and Return the Final Model ---
    dp_model = keras.Model(
        inputs=[word_input, left_context_input, right_context_input, lc_arc_label_input],
        outputs=arc_label_output, name='SPUContext_DP_Model'
    )
    logger.info("SPUContext DP model created successfully.")
    return dp_model

def process_dp_input(
    word_index: int,
    sentence_tokens: List[str],
    spu_tokenizer_word: 'spm.SentencePieceProcessor',
    dp_label_tokenizer: tf.keras.preprocessing.text.Tokenizer,
    arc_label_vector_len: int,
    previous_arcs: List[int],
    previous_labels: List[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares input arrays for a single word for the DP model.

    Args:
        word_index (int): Index of the current word.
        sentence_tokens (List[str]): All tokens in the sentence.
        spu_tokenizer_word: The SentencePiece tokenizer.
        dp_label_tokenizer: The Keras tokenizer for DP labels.
        arc_label_vector_len (int): Total length of the arc+label vector.
        previous_arcs (List[int]): History of predicted arc indices.
        previous_labels (List[int]): History of predicted label indices.

    Returns:
        A tuple of NumPy arrays ready for model input:
        (current_word, left_context, right_context, left_arc_label_history)
    """
    label_vocab_size = len(dp_label_tokenizer.word_index) + 1

    # 1. Process word and its context    
    current_word, left_context, right_context = process_word_context(
        word_index, sentence_tokens, spu_tokenizer_word, SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN
    )

    # 2. Create the history of previous arc-label predictions
    left_arc_label_history = np.zeros((SENTENCE_MAX_LEN, arc_label_vector_len), dtype=np.float32)
    for prev_idx in range(word_index):
        arc = previous_arcs[prev_idx]
        label = previous_labels[prev_idx]
        arc_vector = tf.keras.utils.to_categorical(arc, num_classes=SENTENCE_MAX_LEN + 1)
        label_vector = tf.keras.utils.to_categorical(label, num_classes=label_vocab_size)
        combined_vector = np.concatenate([arc_vector, label_vector])

        # Ensure vector length matches expected input shape
        if len(combined_vector) != arc_label_vector_len:
            raise ValueError(f"Generated arc-label vector length ({len(combined_vector)}) != expected length ({arc_label_vector_len}).")
        position = (SENTENCE_MAX_LEN - word_index) + prev_idx
        left_arc_label_history[position] = combined_vector

    # 3. Expand dims to create a batch of 1    
    return (
        np.expand_dims(current_word, axis=0),
        np.expand_dims(left_context, axis=0),
        np.expand_dims(right_context, axis=0),
        np.expand_dims(left_arc_label_history, axis=0)
    )

# --- General DP Utilities (Consolidated) ---

def decode_arc_label_vector(
    vector: np.ndarray, sentence_max_len: int, label_vocab_size: int
) -> Tuple[int, int]:
    """Decodes the model's raw output vector into a predicted arc and label."""
    arc_logits = vector[:sentence_max_len + 1]
    arc = np.argmax(arc_logits, axis=-1)
    label_start_index = sentence_max_len + 1
    label_end_index = label_start_index + label_vocab_size + 1
    label_logits = vector[label_start_index:label_end_index]
    label = np.argmax(label_logits, axis=-1)
    return int(arc), int(label)

def dp_pos_to_displacy_format(
    dp_result: List[Tuple[int, str, int, str]],
    pos_result: List[Tuple[str, str]] = None
) -> List[Dict]:
    """Converts dependency parsing results to the spacy.displacy format."""
    if pos_result is None:
        pos_result = [(res[1], 'X') for res in dp_result]

    if len(dp_result) != len(pos_result) or any(d[1] != p[0] for d, p in zip(dp_result, pos_result)):
        logger.warning("DP and POS token mismatch. Displacy output may be incorrect.")
        words_data = [{'text': res[1], 'tag': 'X'} for res in dp_result]
    else:
        words_data = [{'text': token, 'tag': pos_tag} for (token, pos_tag) in pos_result]

    arcs_data = []
    for word_idx, _, head_idx, dep_label in dp_result:
        if head_idx <= 0:
            continue
        start, end = word_idx - 1, head_idx - 1
        direction = 'left' if start > end else 'right'
        if direction == 'left':
            start, end = end, start
        arcs_data.append({'start': start, 'end': end, 'label': dep_label, 'dir': direction})
    
    return [{'words': words_data, 'arcs': arcs_data}]