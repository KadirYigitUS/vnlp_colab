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
Keras 3 compliant utilities for the Stemmer/Morphological Analyzer.

This module provides the modernized model creation and data processing functions
for the morphological disambiguation model.
"""
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)

def create_stemmer_model(
    num_max_analysis: int,
    stem_max_len: int,
    char_vocab_size: int,
    char_embed_size: int,
    stem_num_rnn_units: int,
    tag_max_len: int,
    tag_vocab_size: int,
    tag_embed_size: int,
    tag_num_rnn_units: int,
    sentence_max_len: int,
    surface_token_max_len: int,
    embed_join_type: str = 'add',
    dropout: float = 0.2,
    num_rnn_stacks: int = 1
) -> keras.Model:
    """
    Builds the stemmer/morphological disambiguation model using Keras 3 Functional API.

    This architecture is a 1:1 replica of the original model blueprint.

    Args:
        num_max_analysis (int): Max number of morphological analyses per token.
        stem_max_len (int): Max length of a stem in characters.
        char_vocab_size (int): Vocabulary size for characters.
        char_embed_size (int): Dimension of character embeddings.
        stem_num_rnn_units (int): Number of units in the stem processing GRU.
        tag_max_len (int): Max length of a tag sequence.
        tag_vocab_size (int): Vocabulary size for morphological tags.
        tag_embed_size (int): Dimension of tag embeddings.
        tag_num_rnn_units (int): Number of units in the tag processing GRU.
        sentence_max_len (int): Max number of tokens in the context window.
        surface_token_max_len (int): Max length of a surface token in characters.
        embed_join_type (str): How to join stem and tag embeddings ('add' or 'concat').
        dropout (float): Dropout rate.
        num_rnn_stacks (int): Number of layers in RNN stacks.

    Returns:
        keras.Model: The compiled Keras model for morphological disambiguation.
    """
    logger.info("Creating Keras 3 Stemmer/Morphological Disambiguation model...")
    surface_num_rnn_units = stem_num_rnn_units + tag_num_rnn_units

    # --- 1. Define Inputs ---
    stem_input = keras.Input(shape=(num_max_analysis, stem_max_len), dtype='int32', name='stem_input')
    tag_input = keras.Input(shape=(num_max_analysis, tag_max_len), dtype='int32', name='tag_input')
    surface_left_input = keras.Input(shape=(sentence_max_len, surface_token_max_len), dtype='int32', name='surface_left_input')
    surface_right_input = keras.Input(shape=(sentence_max_len, surface_token_max_len), dtype='int32', name='surface_right_input')

    # --- 2. Define Shared Layers & Sub-Models ---
    char_embedding = keras.layers.Embedding(char_vocab_size, char_embed_size, name='char_embedding')
    tag_embedding = keras.layers.Embedding(tag_vocab_size, tag_embed_size, name='tag_embedding')

    # Stem processing sub-model
    stem_rnn_layers = []
    for _ in range(num_rnn_stacks - 1):
        stem_rnn_layers.append(keras.layers.Bidirectional(keras.layers.GRU(stem_num_rnn_units, return_sequences=True)))
        stem_rnn_layers.append(keras.layers.Dropout(dropout))
    stem_rnn_layers.append(keras.layers.Bidirectional(keras.layers.GRU(stem_num_rnn_units)))
    stem_rnn_layers.append(keras.layers.Dropout(dropout))
    stem_rnn = keras.Sequential(stem_rnn_layers, name='stem_char_rnn')

    # Tag processing sub-model
    tag_rnn_layers = []
    for _ in range(num_rnn_stacks - 1):
        tag_rnn_layers.append(keras.layers.Bidirectional(keras.layers.GRU(tag_num_rnn_units, return_sequences=True)))
        tag_rnn_layers.append(keras.layers.Dropout(dropout))
    tag_rnn_layers.append(keras.layers.Bidirectional(keras.layers.GRU(tag_num_rnn_units)))
    tag_rnn_layers.append(keras.layers.Dropout(dropout))
    tag_rnn = keras.Sequential(tag_rnn_layers, name='tag_char_rnn')

    # Surface form processing sub-model (shared for left/right context)
    surface_rnn_layers = []
    for _ in range(num_rnn_stacks - 1):
        surface_rnn_layers.append(keras.layers.Bidirectional(keras.layers.GRU(surface_num_rnn_units, return_sequences=True)))
        surface_rnn_layers.append(keras.layers.Dropout(dropout))
    surface_rnn_layers.append(keras.layers.Bidirectional(keras.layers.GRU(surface_num_rnn_units)))
    surface_rnn_layers.append(keras.layers.Dropout(dropout))
    surface_rnn = keras.Sequential(surface_rnn_layers, name='surface_char_rnn')

    # --- 3. Build "R" Component (Analysis Representation) ---
    stem_embedded = char_embedding(stem_input)
    td_stem_rnn = keras.layers.TimeDistributed(stem_rnn)(stem_embedded)

    tag_embedded = tag_embedding(tag_input)
    td_tag_rnn = keras.layers.TimeDistributed(tag_rnn)(tag_embedded)

    if embed_join_type == 'add':
        joined_stem_tag = keras.layers.Add()([td_stem_rnn, td_tag_rnn])
    else: # 'concat'
        joined_stem_tag = keras.layers.Concatenate()([td_stem_rnn, td_tag_rnn])
    R = keras.layers.Activation('tanh', name='analysis_representation')(joined_stem_tag)

    # --- 4. Build "h" Component (Context Representation) ---
    surface_embedded_left = char_embedding(surface_left_input)
    td_surface_left = keras.layers.TimeDistributed(surface_rnn)(surface_embedded_left)
    surface_left_context = keras.layers.GRU(surface_num_rnn_units, name='left_context_gru')(td_surface_left)

    surface_embedded_right = char_embedding(surface_right_input)
    td_surface_right = keras.layers.TimeDistributed(surface_rnn)(surface_embedded_right)
    surface_right_context = keras.layers.GRU(surface_num_rnn_units, go_backwards=True, name='right_context_gru')(td_surface_right)

    if embed_join_type == 'add':
        joined_context = keras.layers.Add()([surface_left_context, surface_right_context])
    else: # 'concat'
        joined_context = keras.layers.Concatenate()([surface_left_context, surface_right_context])
    h = keras.layers.Activation('tanh', name='context_representation')(joined_context)

    # --- 5. Final Combination and Output ---
    p = keras.layers.Dot(axes=(2, 1))([R, h])
    p = keras.layers.Dense(num_max_analysis * 2, activation='tanh')(p)
    p = keras.layers.Dropout(dropout)(p)
    p = keras.layers.Dense(num_max_analysis, activation='softmax', name='disambiguation_output')(p)

    model = keras.Model(
        inputs=[stem_input, tag_input, surface_left_input, surface_right_input],
        outputs=p,
        name='StemmerDisambiguationModel'
    )
    logger.info("Stemmer/Morphological Disambiguation model created successfully.")
    return model

def process_stemmer_input(
    data: List[Tuple[List[str], List[List[Dict[str, Any]]]]],
    tokenizer_char: tf.keras.preprocessing.text.Tokenizer,
    tokenizer_tag: tf.keras.preprocessing.text.Tokenizer,
    stem_max_len: int,
    tag_max_len: int,
    surface_token_max_len: int,
    sentence_max_len: int,
    num_max_analysis: int
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """
    Processes raw data into NumPy arrays for the stemmer model.

    Args:
        data: A list of sentences, where each sentence is a tuple containing
              a list of surface tokens and a list of analysis candidates for each token.
        tokenizer_char: Keras tokenizer for characters.
        tokenizer_tag: Keras tokenizer for morphological tags.
        All other args are model hyperparameters for padding and truncation.

    Returns:
        A tuple containing:
        - A tuple of the four input NumPy arrays for the model.
        - A NumPy array of labels (for training, not used in inference).
    """
    all_tokens_data = []
    for sentence_tokens, sentence_analyses in data:
        for i in range(len(sentence_tokens)):
            all_tokens_data.append({
                "analyses": sentence_analyses[i],
                "left_context": sentence_tokens[max(0, i - sentence_max_len):i],
                "right_context": sentence_tokens[i + 1 : i + 1 + sentence_max_len]
            })

    num_total_tokens = len(all_tokens_data)
    if num_total_tokens == 0:
        empty_inputs = (np.array([]), np.array([]), np.array([]), np.array([]))
        empty_labels = np.array([])
        return empty_inputs, empty_labels

    # Pre-allocate NumPy arrays
    stems_batch = np.zeros((num_total_tokens, num_max_analysis, stem_max_len), dtype=np.int32)
    tags_batch = np.zeros((num_total_tokens, num_max_analysis, tag_max_len), dtype=np.int32)
    labels_batch = np.zeros((num_total_tokens, num_max_analysis), dtype=np.int32)
    left_context_batch = np.zeros((num_total_tokens, sentence_max_len, surface_token_max_len), dtype=np.int32)
    right_context_batch = np.zeros((num_total_tokens, sentence_max_len, surface_token_max_len), dtype=np.int32)

    for i, token_data in enumerate(all_tokens_data):
        stem_candidates = [analysis[0] for analysis in token_data["analyses"]]
        tag_candidates_as_lists = [analysis[2] for analysis in token_data["analyses"]]
        
        # Join tags into space-separated strings for Keras tokenizer
        tag_candidates_as_strings = [' '.join(tags) for tags in tag_candidates_as_lists]

        # Tokenize and pad stems
        tokenized_stems = tokenizer_char.texts_to_sequences(stem_candidates)
        padded_stems = keras.preprocessing.sequence.pad_sequences(
            tokenized_stems, maxlen=stem_max_len, padding='pre', truncating='pre'
        )

        # Tokenize and pad tags
        tokenized_tags = tokenizer_tag.texts_to_sequences(tag_candidates_as_strings)
        padded_tags = keras.preprocessing.sequence.pad_sequences(
            tokenized_tags, maxlen=tag_max_len, padding='pre', truncating='pre'
        )
        
        num_analyses = min(len(padded_stems), num_max_analysis)
        stems_batch[i, :num_analyses] = padded_stems[:num_analyses]
        tags_batch[i, :num_analyses] = padded_tags[:num_analyses]
        
        # Create labels (1 for the first/correct analysis, 0 otherwise)
        if num_analyses > 0:
            labels_batch[i, 0] = 1

        # Process left context
        if token_data["left_context"]:
            tokenized_left = tokenizer_char.texts_to_sequences(token_data["left_context"])
            padded_left = keras.preprocessing.sequence.pad_sequences(
                tokenized_left, maxlen=surface_token_max_len, padding='pre', truncating='pre'
            )
            left_context_batch[i, -len(padded_left):] = padded_left
        
        # Process right context
        if token_data["right_context"]:
            tokenized_right = tokenizer_char.texts_to_sequences(token_data["right_context"])
            padded_right = keras.preprocessing.sequence.pad_sequences(
                tokenized_right, maxlen=surface_token_max_len, padding='pre', truncating='post'
            )
            right_context_batch[i, :len(padded_right)] = padded_right

    inputs = (stems_batch, tags_batch, left_context_batch, right_context_batch)
    return inputs, labels_batch