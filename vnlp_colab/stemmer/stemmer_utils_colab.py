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
for the morphological disambiguation model. THIS IS THE FINAL BLUEPRINT-ACCURATE
VERSION that precisely matches the original saved model architecture.
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
    Builds the stemmer model using the Keras 3 Functional API.

    This architecture is a 1:1 replica of the `stemmer_morph_analyzer_model_architecture.txt`
    blueprint, ensuring perfect weight compatibility.
    """
    logger.info("Creating Keras 3 Stemmer model (Blueprint-Accurate Replica)...")

    # --- 1. Define Inputs matching the blueprint ---
    stem_input = keras.Input(shape=(num_max_analysis, stem_max_len), dtype='int32', name='input_layer')
    surface_left_input = keras.Input(shape=(sentence_max_len, surface_token_max_len), dtype='int32', name='input_layer_4')
    surface_right_input = keras.Input(shape=(sentence_max_len, surface_token_max_len), dtype='int32', name='input_layer_6')
    tag_input = keras.Input(shape=(num_max_analysis, tag_max_len), dtype='int32', name='input_layer_2')

    # --- 2. Define Shared Embedding Layers as per the blueprint ---
    char_embedding_layer = keras.layers.Embedding(char_vocab_size, char_embed_size, name='embedding')
    tag_embedding_layer = keras.layers.Embedding(tag_vocab_size, tag_embed_size, name='embedding_1')

    # --- 3. Apply Embeddings ---
    # The blueprint shows one embedding layer is shared across three inputs
    stem_embedded = char_embedding_layer(stem_input)
    left_context_embedded = char_embedding_layer(surface_left_input)
    right_context_embedded = char_embedding_layer(surface_right_input)
    tag_embedded = tag_embedding_layer(tag_input)

    # --- 4. Reshape Embeddings to Flatten the character dimension before Dense layers ---
    # This step is implicitly required to match the blueprint's TimeDistributed input shapes
    stem_flat = keras.layers.Reshape((num_max_analysis, -1))(stem_embedded)
    tag_flat = keras.layers.Reshape((num_max_analysis, -1))(tag_embedded)
    left_context_flat = keras.layers.Reshape((sentence_max_len, -1))(left_context_embedded)
    right_context_flat = keras.layers.Reshape((sentence_max_len, -1))(right_context_embedded)

    # --- 5. Build the Model Graph exactly as per the Blueprint ---
    td_stem = keras.layers.TimeDistributed(keras.layers.Dense(256), name='time_distributed')(stem_flat)
    td_tag = keras.layers.TimeDistributed(keras.layers.Dense(256), name='time_distributed_1')(tag_flat)
    
    td_left = keras.layers.TimeDistributed(keras.layers.Dense(512), name='time_distributed_2')(left_context_flat)
    td_right = keras.layers.TimeDistributed(keras.layers.Dense(512), name='time_distributed_3')(right_context_flat)
    
    left_context_gru = keras.layers.GRU(256, name='gru_3')(td_left)
    right_context_gru = keras.layers.GRU(256, name='gru_5')(td_right)

    added_stem_tag = keras.layers.Add(name='add')([td_stem, td_tag])
    added_context = keras.layers.Add(name='add_1')([left_context_gru, right_context_gru])

    R = keras.layers.Activation('tanh', name='activation')(added_stem_tag)
    h = keras.layers.Activation('tanh', name='activation_1')(added_context)

    dot_product = keras.layers.Dot(axes=(2, 1), name='dot')([R, h])

    dense_out_1 = keras.layers.Dense(20, name='dense')(dot_product)
    dropout_out = keras.layers.Dropout(dropout, name='dropout_4')(dense_out_1)
    final_output = keras.layers.Dense(num_max_analysis, activation='softmax', name='dense_1')(dropout_out)

    model = keras.Model(
        inputs=[stem_input, surface_left_input, surface_right_input, tag_input],
        outputs=final_output,
        name='StemmerDisambiguationModel_Blueprint_Corrected'
    )
    
    logger.info("Stemmer model (Blueprint-Accurate Replica) created successfully.")
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