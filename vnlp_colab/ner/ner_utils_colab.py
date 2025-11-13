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
Keras 3 compliant utilities for the SPUContext Named Entity Recognizer (NER).

This module provides the modernized model creation function for the NER model,
ensuring compatibility with the latest TensorFlow/Keras versions and Colab.
"""
import logging
import re
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

# --- SPUContextNER Utilities ---

def create_spucontext_ner_model(
    vocab_size: int,
    entity_vocab_size: int,
    word_embedding_dim: int,
    word_embedding_matrix: np.ndarray,
    num_rnn_units: int,
    num_rnn_stacks: int,
    fc_units_multiplier: tuple[int, int],
    dropout: float
) -> keras.Model:
    """
    Builds the SPUContext NER model using the Keras 3 Functional API.

    This architecture is a 1:1 replica of the original model blueprint, ensuring
    weight compatibility while adhering to modern Keras standards.

    Args:
        vocab_size (int): Vocabulary size for the word embedding layer.
        entity_vocab_size (int): Vocabulary size for the NER entity tags.
            The output layer will have `entity_vocab_size + 1` units.
        word_embedding_dim (int): Dimension of the word embeddings.
        word_embedding_matrix (np.ndarray): Pre-trained word embedding matrix.
        num_rnn_units (int): Number of units in the GRU layers.
        num_rnn_stacks (int): Number of layers in each RNN stack.
        fc_units_multiplier (tuple[int, int]): Multipliers for the dense layers.
        dropout (float): Dropout rate.

    Returns:
        keras.Model: The compiled Keras model.
    """
    logger.info("Creating Keras 3 SPUContext NER model...")

    # --- 1. Define Functional API Inputs ---
    word_input = keras.Input(shape=(TOKEN_PIECE_MAX_LEN,), name='word_input', dtype='int32')
    left_context_input = keras.Input(shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), name='left_context_input', dtype='int32')
    right_context_input = keras.Input(shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), name='right_context_input', dtype='int32')
    lc_entity_input = keras.Input(shape=(SENTENCE_MAX_LEN, entity_vocab_size + 1), name='lc_entity_input', dtype='float32')

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

    # Branch 4: Previously predicted (left) NER tags
    lc_entity_stack = create_rnn_stacks(num_rnn_stacks, num_rnn_units, dropout)
    lc_entity_output = lc_entity_stack(lc_entity_input)

    # --- 4. Concatenate Branches and Add Final FC Layers ---
    concatenated = keras.layers.Concatenate()([word_output, left_context_output, right_context_output, lc_entity_output])

    x = keras.layers.Dense(num_rnn_units * fc_units_multiplier[0], activation='relu')(concatenated)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(num_rnn_units * fc_units_multiplier[1], activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    entity_output = keras.layers.Dense(entity_vocab_size + 1, activation='softmax')(x)
    
    # --- 5. Build and Return the Final Model ---
    ner_model = keras.Model(
        inputs=[word_input, left_context_input, right_context_input, lc_entity_input],
        outputs=entity_output, name='SPUContext_NER_Model'
    )
    logger.info("SPUContext NER model created successfully.")
    return ner_model

def process_ner_input(
    word_index: int,
    sentence_tokens: List[str],
    spu_tokenizer_word: 'spm.SentencePieceProcessor',
    ner_label_tokenizer: tf.keras.preprocessing.text.Tokenizer,
    previous_predictions: List[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares input arrays for a single word for the NER model.

    This function vectorizes the word and its context, as well as the history of
    previously predicted entity tags for the autoregressive loop.

    Args:
        word_index (int): Index of the current word in the sentence.
        sentence_tokens (List[str]): List of all tokens in the sentence.
        spu_tokenizer_word: The SentencePiece tokenizer.
        ner_label_tokenizer: The Keras tokenizer for NER labels.
        previous_predictions (List[int]): A list of integer NER tag predictions
            for the preceding words in the sentence.

    Returns:
        A tuple of NumPy arrays ready for model input:
        (current_word, left_context, right_context, left_entity_history)
    """
    entity_vocab_size = len(ner_label_tokenizer.word_index) + 1

    # 1. Process word and its context
    current_word, left_context, right_context = process_word_context(
        word_index, sentence_tokens, spu_tokenizer_word, SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN
    )
    
    # 2. Create the history of previous NER tag predictions    
    left_entity_history = np.zeros((SENTENCE_MAX_LEN, entity_vocab_size), dtype=np.float32)
    if word_index > 0:
        one_hot_preds = tf.keras.utils.to_categorical(previous_predictions, num_classes=entity_vocab_size)
        start_pos = SENTENCE_MAX_LEN - word_index
        left_entity_history[start_pos : start_pos + len(one_hot_preds)] = one_hot_preds
    
    # 3. Expand dims to create a batch of 1 for model prediction
    return (
        np.expand_dims(current_word, axis=0),
        np.expand_dims(left_context, axis=0),
        np.expand_dims(right_context, axis=0),
        np.expand_dims(left_entity_history, axis=0)
    )

# --- CharNER Utilities (Consolidated) ---

def create_charner_model(
    char_vocab_size: int, embed_size: int, seq_len_max: int,
    num_rnn_stacks: int, rnn_dim: int, mlp_dim: int,
    num_classes: int, dropout: float
) -> keras.Model:
    """Builds the Character-Level NER model."""
    logger.info("Creating Keras 3 CharNER model...")
    model = keras.Sequential(name="CharNER_Model")
    model.add(keras.layers.Input(shape=(seq_len_max,), dtype='int32'))
    model.add(keras.layers.Embedding(input_dim=char_vocab_size, output_dim=embed_size))

    for _ in range(num_rnn_stacks):
        model.add(keras.layers.Bidirectional(keras.layers.GRU(rnn_dim, return_sequences=True)))
        model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(mlp_dim, activation='relu'))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    logger.info("CharNER model created successfully.")
    return model

def ner_to_displacy_format(text: str, ner_result: List[Tuple[str, str]]) -> Dict:
    """Converts NER results to the dictionary format for spacy.displacy."""
    displacy_data = {'text': text, 'ents': [], 'title': None}
    current_entity = None
    entity_text = ""
    start_char = 0

    # Find character spans for each token
    token_spans = []
    search_start = 0
    for token, _ in ner_result:
        start = text.find(token, search_start)
        if start != -1:
            end = start + len(token)
            token_spans.append((start, end))
            search_start = end
        else:
            token_spans.append(None) # Could not find token

    # Group consecutive entities
    for i, ((token, entity), span) in enumerate(zip(ner_result, token_spans)):
        if span is None: continue

        if entity != 'O':
            if current_entity is None:
                # Start of a new entity
                current_entity = entity
                entity_text = token
                start_char = span[0]
            elif entity == current_entity:
                # Continuation of the current entity
                entity_text += text[token_spans[i-1][1]:span[0]] + token # Add whitespace and token
            else:
                # End of the previous entity, start of a new one
                end_char = token_spans[i-1][1]
                displacy_data['ents'].append({'start': start_char, 'end': end_char, 'label': current_entity})
                current_entity = entity
                entity_text = token
                start_char = span[0]
        elif current_entity is not None:
            # End of an entity span
            end_char = token_spans[i-1][1]
            displacy_data['ents'].append({'start': start_char, 'end': end_char, 'label': current_entity})
            current_entity = None
            entity_text = ""

    # Add the last entity if the sentence ends with one
    if current_entity is not None:
        end_char = token_spans[-1][1]
        displacy_data['ents'].append({'start': start_char, 'end': end_char, 'label': current_entity})

    return displacy_data