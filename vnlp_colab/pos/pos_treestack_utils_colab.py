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
Keras 3 compliant utilities for the TreeStack Part-of-Speech (PoS) Tagger.

This module provides the modernized model creation and data processing functions
for the TreeStack PoS tagger, ensuring compatibility with Keras 3 and Colab.
"""
import logging
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


logger = logging.getLogger(__name__)


def create_treestack_pos_model(
    word_embedding_vocab_size: int,
    word_embedding_vector_size: int,
    word_embedding_matrix: np.ndarray,
    pos_vocab_size: int,
    sentence_max_len: int,
    tag_max_len: int,
    num_rnn_stacks: int,
    tag_num_rnn_units: int,
    lc_num_rnn_units: int,
    rc_num_rnn_units: int,
    dropout: float,
    tag_embedding_matrix: np.ndarray,
    fc_units_multipliers: Tuple[int, int],
) -> keras.Model:
    """
    Builds the TreeStack PoS tagger model using the Keras 3 Functional API.

    This architecture is a 1:1 replica of the original model, ensuring
    weight compatibility while adhering to modern Keras standards.
    """
    logger.info("Creating Keras 3 TreeStack PoS Tagger model...")

    tag_vocab_size, tag_embed_size = tag_embedding_matrix.shape

    # --- 1. Define Shared Layers & Sub-Models ---
    word_embedding_layer = keras.layers.Embedding(
        input_dim=word_embedding_vocab_size,
        output_dim=word_embedding_vector_size,
        embeddings_initializer=keras.initializers.Constant(word_embedding_matrix),
        trainable=False,
        name="word_embedding",
    )
    tag_embedding_layer = keras.layers.Embedding(
        input_dim=tag_vocab_size,
        output_dim=tag_embed_size,
        weights=[tag_embedding_matrix],
        trainable=False,
        name="tag_embedding",
    )

    tag_rnn = keras.Sequential(name="tag_rnn")
    for _ in range(num_rnn_stacks - 1):
        tag_rnn.add(keras.layers.GRU(tag_num_rnn_units, return_sequences=True))
        tag_rnn.add(keras.layers.Dropout(dropout))
    tag_rnn.add(keras.layers.GRU(tag_num_rnn_units))
    tag_rnn.add(keras.layers.Dropout(dropout))

    lc_rnn = keras.Sequential(name="left_context_rnn")
    for _ in range(num_rnn_stacks - 1):
        lc_rnn.add(keras.layers.GRU(lc_num_rnn_units, return_sequences=True))
        lc_rnn.add(keras.layers.Dropout(dropout))
    lc_rnn.add(keras.layers.GRU(lc_num_rnn_units))
    lc_rnn.add(keras.layers.Dropout(dropout))

    rc_rnn = keras.Sequential(name="right_context_rnn")
    for _ in range(num_rnn_stacks - 1):
        rc_rnn.add(keras.layers.GRU(rc_num_rnn_units, return_sequences=True, go_backwards=True))
        rc_rnn.add(keras.layers.Dropout(dropout))
    rc_rnn.add(keras.layers.GRU(rc_num_rnn_units, go_backwards=True))
    rc_rnn.add(keras.layers.Dropout(dropout))

    # --- 2. Define Functional API Inputs ---
    word_input = keras.Input(shape=(1,), name="word_input", dtype="int32")
    tag_input = keras.Input(shape=(tag_max_len,), name="tag_input", dtype="int32")
    lc_word_input = keras.Input(shape=(sentence_max_len,), name="lc_word_input", dtype="int32")
    lc_tag_input = keras.Input(shape=(sentence_max_len, tag_max_len), name="lc_tag_input", dtype="int32")
    lc_pos_input = keras.Input(shape=(sentence_max_len, pos_vocab_size + 1), name="lc_pos_input", dtype="float32")
    rc_word_input = keras.Input(shape=(sentence_max_len,), name="rc_word_input", dtype="int32")
    rc_tag_input = keras.Input(shape=(sentence_max_len, tag_max_len), name="rc_tag_input", dtype="int32")

    # --- 3. Build the Graph ---
    # Current word branch
    word_embedded = keras.layers.Flatten()(word_embedding_layer(word_input))
    tag_embedded = tag_embedding_layer(tag_input)
    tag_rnn_output = tag_rnn(tag_embedded)
    current_word_vector = keras.layers.Concatenate()([word_embedded, tag_rnn_output])

    # Left context branch
    lc_word_embedded = word_embedding_layer(lc_word_input)
    lc_tag_embedded = tag_embedding_layer(lc_tag_input)
    lc_td_tag_rnn_output = keras.layers.TimeDistributed(tag_rnn)(lc_tag_embedded)
    lc_combined = keras.layers.Concatenate()([lc_word_embedded, lc_td_tag_rnn_output, lc_pos_input])
    lc_output = lc_rnn(lc_combined)

    # Right context branch
    rc_word_embedded = word_embedding_layer(rc_word_input)
    rc_tag_embedded = tag_embedding_layer(rc_tag_input)
    rc_td_tag_rnn_output = keras.layers.TimeDistributed(tag_rnn)(rc_tag_embedded)
    rc_combined = keras.layers.Concatenate()([rc_word_embedded, rc_td_tag_rnn_output])
    rc_output = rc_rnn(rc_combined)

    # --- 4. Concatenate and Final FC Layers ---
    concatenated = keras.layers.Concatenate()([current_word_vector, lc_output, rc_output])
    x = keras.layers.Dense(tag_num_rnn_units * fc_units_multipliers[0], activation="relu")(concatenated)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(tag_num_rnn_units * fc_units_multipliers[1], activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    pos_output = keras.layers.Dense(pos_vocab_size + 1, activation="sigmoid")(x)

    model = keras.Model(
        inputs=[word_input, tag_input, lc_word_input, lc_tag_input, lc_pos_input, rc_word_input, rc_tag_input],
        outputs=pos_output,
        name="TreeStack_PoS_Model",
    )
    logger.info("TreeStack PoS Tagger model created successfully.")
    return model


def process_treestack_pos_input(
    tokens: List[str],
    sentence_analyses: List[str],
    previous_pos_preds: List[int],
    tokenizer_word: tf.keras.preprocessing.text.Tokenizer,
    tokenizer_morph_tag: tf.keras.preprocessing.text.Tokenizer,
    tokenizer_pos_label: tf.keras.preprocessing.text.Tokenizer,
    word_form: str,
    sentence_max_len: int,
    tag_max_len: int,
) -> Tuple[np.ndarray, ...]:
    """
    Prepares a batch of inputs for a single token for the TreeStack PoS model.
    """
    stems = [analysis.split("+")[0] for analysis in sentence_analyses]
    pos_vocab_size = len(tokenizer_pos_label.word_index)
    pos_vector_size = pos_vocab_size + 1
    
    # Determine which word form to use (whole vs. stem)
    words_to_tokenize = []
    if word_form == 'whole':
        for whole, stem in zip(tokens, stems):
            if not tokenizer_word.texts_to_sequences([whole])[0] and tokenizer_word.texts_to_sequences([stem])[0]:
                words_to_tokenize.append(stem)
            else:
                words_to_tokenize.append(whole)
    else: # 'stem'
        for whole, stem in zip(tokens, stems):
            if not tokenizer_word.texts_to_sequences([stem])[0] and tokenizer_word.texts_to_sequences([whole])[0]:
                words_to_tokenize.append(whole)
            else:
                words_to_tokenize.append(stem)

    # --- Prepare inputs for the current token (t) ---
    t = len(previous_pos_preds)
    
    # 1. Current word, tag, and pos
    word_t = tokenizer_word.texts_to_sequences([[words_to_tokenize[t]]])[0]
    tags_t_str = " ".join(sentence_analyses[t].split("+")[1:])
    tags_t = tokenizer_morph_tag.texts_to_sequences([tags_t_str])
    tags_t = keras.preprocessing.sequence.pad_sequences(tags_t, maxlen=tag_max_len, padding="pre")[0]

    # 2. Left Context
    lc_words_str = words_to_tokenize[:t]
    lc_words = tokenizer_word.texts_to_sequences([lc_words_str])[0]
    lc_words = keras.preprocessing.sequence.pad_sequences([lc_words], maxlen=sentence_max_len, padding="pre")[0]
    
    lc_tags_str = [" ".join(a.split("+")[1:]) for a in sentence_analyses[:t]]
    lc_tags_seq = tokenizer_morph_tag.texts_to_sequences(lc_tags_str)
    lc_tags = keras.preprocessing.sequence.pad_sequences(lc_tags_seq, maxlen=tag_max_len, padding="pre")
    
    # Pad the sequence of tag vectors
    lc_tags_padded = np.zeros((sentence_max_len, tag_max_len), dtype=np.int32)
    if lc_tags.shape[0] > 0:
        lc_tags_padded[-lc_tags.shape[0]:] = lc_tags

    lc_pos_vectors = np.zeros((sentence_max_len, pos_vector_size), dtype=np.float32)
    if previous_pos_preds:
        one_hot_pos = tf.keras.utils.to_categorical(previous_pos_preds, num_classes=pos_vector_size)
        lc_pos_vectors[-len(one_hot_pos):] = one_hot_pos

    # 3. Right Context
    rc_words_str = words_to_tokenize[t + 1:]
    rc_words = tokenizer_word.texts_to_sequences([rc_words_str])[0]
    rc_words = keras.preprocessing.sequence.pad_sequences([rc_words], maxlen=sentence_max_len, padding="post", truncating="post")[0]
    
    rc_tags_str = [" ".join(a.split("+")[1:]) for a in sentence_analyses[t + 1:]]
    rc_tags_seq = tokenizer_morph_tag.texts_to_sequences(rc_tags_str)
    rc_tags = keras.preprocessing.sequence.pad_sequences(rc_tags_seq, maxlen=tag_max_len, padding="pre")
    
    rc_tags_padded = np.zeros((sentence_max_len, tag_max_len), dtype=np.int32)
    if rc_tags.shape[0] > 0:
        rc_tags_padded[:rc_tags.shape[0]] = rc_tags
    
    # 4. Reshape for a single batch item
    return (
        np.array(word_t).reshape(1, 1),
        tags_t.reshape(1, tag_max_len),
        lc_words.reshape(1, sentence_max_len),
        lc_tags_padded.reshape(1, sentence_max_len, tag_max_len),
        lc_pos_vectors.reshape(1, sentence_max_len, pos_vector_size),
        rc_words.reshape(1, sentence_max_len),
        rc_tags_padded.reshape(1, sentence_max_len, tag_max_len),
    )