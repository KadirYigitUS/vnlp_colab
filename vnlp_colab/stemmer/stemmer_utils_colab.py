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

THIS IS THE FINAL, CORRECTED VERSION. It abandons the incorrect blueprint and
builds the true architecture based on the original code's logic, which matches
the 36 weight tensors in the saved weights file.
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
    num_rnn_stacks: int = 1,
    **kwargs # Accept other params to avoid breaking the call
) -> keras.Model:
    """
    Builds the stemmer model using the Keras 3 Functional API. This is the
    correct architecture with 36 weight tensors.
    """
    logger.info("Creating Keras 3 Stemmer model (Final Corrected Architecture)...")
    surface_num_rnn_units = stem_num_rnn_units + tag_num_rnn_units

    # --- 1. Define Inputs ---
    stem_input = keras.Input(shape=(num_max_analysis, stem_max_len), dtype='int32', name='stem_input')
    tag_input = keras.Input(shape=(num_max_analysis, tag_max_len), dtype='int32', name='tag_input')
    surface_left_input = keras.Input(shape=(sentence_max_len, surface_token_max_len), dtype='int32', name='surface_left_input')
    surface_right_input = keras.Input(shape=(sentence_max_len, surface_token_max_len), dtype='int32', name='surface_right_input')

    # --- 2. Define Shared Layers & Reusable Sub-Models ---
    char_embedding = keras.layers.Embedding(char_vocab_size, char_embed_size, name='char_embedding')
    tag_embedding = keras.layers.Embedding(tag_vocab_size, tag_embed_size, name='tag_embedding')

    stem_rnn = keras.Sequential([
        keras.layers.Bidirectional(keras.layers.GRU(stem_num_rnn_units)),
        keras.layers.Dropout(dropout)
    ], name='stem_char_rnn')

    tag_rnn = keras.Sequential([
        keras.layers.Bidirectional(keras.layers.GRU(tag_num_rnn_units)),
        keras.layers.Dropout(dropout)
    ], name='tag_char_rnn')

    # Create two INDEPENDENT RNNs for surface form processing to match the weight count
    surface_char_rnn_left = keras.Sequential([
        keras.layers.Bidirectional(keras.layers.GRU(surface_num_rnn_units)),
        keras.layers.Dropout(dropout)
    ], name='surface_char_rnn_left')

    surface_char_rnn_right = keras.Sequential([
        keras.layers.Bidirectional(keras.layers.GRU(surface_num_rnn_units)),
        keras.layers.Dropout(dropout)
    ], name='surface_char_rnn_right')

    # --- 3. Build "R" Component (Analysis Representation) ---
    stem_embedded = char_embedding(stem_input)
    td_stem_rnn = keras.layers.TimeDistributed(stem_rnn)(stem_embedded)

    tag_embedded = tag_embedding(tag_input)
    td_tag_rnn = keras.layers.TimeDistributed(tag_rnn)(tag_embedded)

    joined_stem_tag = keras.layers.Add()([td_stem_rnn, td_tag_rnn])
    R = keras.layers.Activation('tanh')(joined_stem_tag)

    # --- 4. Build "h" Component (Context Representation) ---
    surface_embedded_left = char_embedding(surface_left_input)
    td_surface_left = keras.layers.TimeDistributed(surface_char_rnn_left)(surface_embedded_left)
    surface_left_context = keras.layers.GRU(surface_num_rnn_units)(td_surface_left)

    surface_embedded_right = char_embedding(surface_right_input)
    td_surface_right = keras.layers.TimeDistributed(surface_char_rnn_right)(surface_embedded_right)
    surface_right_context = keras.layers.GRU(surface_num_rnn_units, go_backwards=True)(td_surface_right)

    joined_context = keras.layers.Add()([surface_left_context, surface_right_context])
    h = keras.layers.Activation('tanh')(joined_context)

    # --- 5. Final Combination and Output ---
    p = keras.layers.Dot(axes=(2, 1))([R, h])
    p = keras.layers.Dense(num_max_analysis * 2, activation='tanh')(p)
    p = keras.layers.Dropout(dropout)(p)
    p = keras.layers.Dense(num_max_analysis, activation='softmax')(p)

    model = keras.Model(
        inputs=[stem_input, tag_input, surface_left_input, surface_right_input],
        outputs=p,
        name='StemmerDisambiguationModel_Final'
    )
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
    """Processes raw data into NumPy arrays for the model."""
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
        return (np.array([]), np.array([]), np.array([]), np.array([])), np.array([])

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