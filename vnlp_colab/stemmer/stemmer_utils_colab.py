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
for the morphological disambiguation model. THIS IS THE CORRECTED VERSION
THAT STRICTLY FOLLOWS THE MODEL BLUEPRINT.
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
    tag_max_len: int,
    tag_vocab_size: int,
    tag_embed_size: int,
    sentence_max_len: int,
    surface_token_max_len: int,
    dropout: float,
    **kwargs # Accept other params to avoid breaking the call
) -> keras.Model:
    """
    Builds the stemmer/morphological disambiguation model using Keras 3 Functional API.

    This architecture is a 1:1 replica of the original model blueprint, ensuring
    weight compatibility. It uses simple GRUs, not Bidirectional GRUs.
    """
    logger.info("Creating Keras 3 Stemmer/Morphological Disambiguation model (Blueprint Replica)...")

    # --- 1. Define Inputs ---
    # These shapes match the blueprint's InputLayer shapes
    stem_input = keras.Input(shape=(num_max_analysis, stem_max_len), dtype='int32', name='input_layer')
    tag_input = keras.Input(shape=(num_max_analysis, tag_max_len), dtype='int32', name='input_layer_2')
    surface_left_input = keras.Input(shape=(sentence_max_len, surface_token_max_len), dtype='int32', name='input_layer_4')
    surface_right_input = keras.Input(shape=(sentence_max_len, surface_token_max_len), dtype='int32', name='input_layer_6')

    # --- 2. Define Shared Embedding Layer ---
    # The blueprint shows a single embedding layer shared across three inputs
    char_embedding_layer = keras.layers.Embedding(char_vocab_size, char_embed_size, name='embedding')
    
    # A separate embedding layer for tags
    tag_embedding_layer = keras.layers.Embedding(tag_vocab_size, tag_embed_size, name='embedding_1')

    # --- 3. Build "R" Component (Analysis Representation) ---
    stem_embedded = char_embedding_layer(stem_input)
    tag_embedded = tag_embedding_layer(tag_input)

    # Flatten the character dimension before the GRU for candidate analyses
    stem_flat = keras.layers.Reshape((num_max_analysis, -1))(stem_embedded)
    tag_flat = keras.layers.Reshape((num_max_analysis, -1))(tag_embedded)
    
    # According to the blueprint, TimeDistributed layers process the embeddings
    td_stem = keras.layers.TimeDistributed(keras.layers.Dense(256, activation='tanh'))(stem_flat)
    td_tag = keras.layers.TimeDistributed(keras.layers.Dense(256, activation='tanh'))(tag_flat)

    joined_stem_tag = keras.layers.Add(name='add')([td_stem, td_tag])
    R = keras.layers.Activation('tanh', name='activation')(joined_stem_tag)

    # --- 4. Build "h" Component (Context Representation) ---
    left_context_embedded = char_embedding_layer(surface_left_input)
    right_context_embedded = char_embedding_layer(surface_right_input)

    # Flatten the character dimension for context
    left_context_flat = keras.layers.Reshape((sentence_max_len, -1))(left_context_embedded)
    right_context_flat = keras.layers.Reshape((sentence_max_len, -1))(right_context_embedded)
    
    td_left = keras.layers.TimeDistributed(keras.layers.Dense(512, activation='tanh'))(left_context_flat)
    td_right = keras.layers.TimeDistributed(keras.layers.Dense(512, activation='tanh'))(right_context_flat)

    # Simple GRUs as per the blueprint
    left_context_vector = keras.layers.GRU(256, name='gru_3')(td_left)
    right_context_vector = keras.layers.GRU(256, go_backwards=True, name='gru_5')(td_right)

    joined_context = keras.layers.Add(name='add_1')([left_context_vector, right_context_vector])
    h = keras.layers.Activation('tanh', name='activation_1')(joined_context)

    # --- 5. Final Combination and Output ---
    p = keras.layers.Dot(axes=(2, 1), name='dot')([R, h])
    p = keras.layers.Dense(20, name='dense')(p)
    p = keras.layers.Dropout(dropout, name='dropout_4')(p)
    p = keras.layers.Dense(num_max_analysis, activation='softmax', name='dense_1')(p)

    model = keras.Model(
        inputs=[stem_input, tag_input, surface_left_input, surface_right_input],
        outputs=p,
        name='StemmerDisambiguationModel_Blueprint'
    )
    logger.info("Stemmer model (Blueprint Replica) created successfully.")
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
    (This function remains correct and does not need changes).
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

    stems_batch = np.zeros((num_total_tokens, num_max_analysis, stem_max_len), dtype=np.int32)
    tags_batch = np.zeros((num_total_tokens, num_max_analysis, tag_max_len), dtype=np.int32)
    labels_batch = np.zeros((num_total_tokens, num_max_analysis), dtype=np.int32)
    left_context_batch = np.zeros((num_total_tokens, sentence_max_len, surface_token_max_len), dtype=np.int32)
    right_context_batch = np.zeros((num_total_tokens, sentence_max_len, surface_token_max_len), dtype=np.int32)

    for i, token_data in enumerate(all_tokens_data):
        stem_candidates = [analysis[0] for analysis in token_data["analyses"]]
        tag_candidates_as_lists = [analysis[2] for analysis in token_data["analyses"]]
        tag_candidates_as_strings = [' '.join(tags) for tags in tag_candidates_as_lists]

        tokenized_stems = tokenizer_char.texts_to_sequences(stem_candidates)
        padded_stems = keras.preprocessing.sequence.pad_sequences(
            tokenized_stems, maxlen=stem_max_len, padding='pre', truncating='pre'
        )

        tokenized_tags = tokenizer_tag.texts_to_sequences(tag_candidates_as_strings)
        padded_tags = keras.preprocessing.sequence.pad_sequences(
            tokenized_tags, maxlen=tag_max_len, padding='pre', truncating='pre'
        )
        
        num_analyses = min(len(padded_stems), num_max_analysis)
        stems_batch[i, :num_analyses] = padded_stems[:num_analyses]
        tags_batch[i, :num_analyses] = padded_tags[:num_analyses]
        
        if num_analyses > 0:
            labels_batch[i, 0] = 1

        if token_data["left_context"]:
            tokenized_left = tokenizer_char.texts_to_sequences(token_data["left_context"])
            padded_left = keras.preprocessing.sequence.pad_sequences(
                tokenized_left, maxlen=surface_token_max_len, padding='pre', truncating='pre'
            )
            left_context_batch[i, -len(padded_left):] = padded_left
        
        if token_data["right_context"]:
            tokenized_right = tokenizer_char.texts_to_sequences(token_data["right_context"])
            padded_right = keras.preprocessing.sequence.pad_sequences(
                tokenized_right, maxlen=surface_token_max_len, padding='pre', truncating='post'
            )
            right_context_batch[i, :len(padded_right)] = padded_right

    inputs = (stems_batch, tags_batch, left_context_batch, right_context_batch)
    return inputs, labels_batch