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
Keras 3 compliant utilities for the SPUCBiGRU Sentiment Analyzer.

This module provides the modernized model creation and data processing functions
for the sentiment analysis model.
"""
import logging
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from tensorflow import keras

if TYPE_CHECKING:
    import sentencepiece as spm

logger = logging.getLogger(__name__)


def create_spucbigru_sentiment_model(
    text_max_len: int,
    vocab_size: int,
    word_embedding_dim: int,
    word_embedding_matrix: np.ndarray,
    num_rnn_units: int,
    num_rnn_stacks: int,
    dropout_rate: float,
) -> keras.Model:
    """
    Creates a Bidirectional GRU model for sentiment analysis using Keras 3 Functional API.

    This architecture is a 1:1 replica of the original model, ensuring
    weight compatibility.

    Args:
        text_max_len (int): Maximum length of the input sequence.
        vocab_size (int): Vocabulary size for the embedding layer.
        word_embedding_dim (int): Dimension of the word embeddings.
        word_embedding_matrix (np.ndarray): Pre-trained word embedding matrix.
        num_rnn_units (int): Number of units in the GRU layers.
        num_rnn_stacks (int): Number of stacked Bidirectional GRU layers.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        keras.Model: The compiled Keras model for sentiment analysis.
    """
    logger.info("Creating Keras 3 SPUCBiGRU Sentiment model...")
    
    # --- 1. Define Input and Embedding Layers ---
    input_layer = keras.Input(shape=(text_max_len,), name="input_layer", dtype="int32")
    embedding_layer = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=word_embedding_dim,
        embeddings_initializer=keras.initializers.Constant(word_embedding_matrix),
        trainable=False,
        name="word_embedding",
    )(input_layer)

    # --- 2. Build Stacked Bidirectional GRU Layers ---
    model_flow = embedding_layer
    for i in range(num_rnn_stacks):
        model_flow = keras.layers.Bidirectional(
            keras.layers.GRU(num_rnn_units, dropout=dropout_rate, return_sequences=True),
            name=f"bidirectional_gru_{i+1}",
        )(model_flow)

    # --- 3. Add Pooling and Final Dense Layers ---
    model_flow = keras.layers.GlobalAveragePooling1D(name="global_avg_pooling")(model_flow)
    model_flow = keras.layers.Dropout(dropout_rate, name="post_pooling_dropout")(model_flow)
    model_flow = keras.layers.Dense(num_rnn_units // 8, activation='relu', name="intermediate_dense")(model_flow)
    model_flow = keras.layers.Dropout(dropout_rate, name="final_dropout")(model_flow)
    output_layer = keras.layers.Dense(1, activation='sigmoid', name="sentiment_output")(model_flow)

    # --- 4. Create and Return the Model ---
    model = keras.models.Model(inputs=input_layer, outputs=output_layer, name="SPUCBiGRU_Sentiment_Model")
    logger.info("SPUCBiGRU Sentiment model created successfully.")
    return model


def process_sentiment_input(
    text: str, tokenizer: "spm.SentencePieceProcessor", text_max_len: int
) -> np.ndarray:
    """
    Tokenizes, truncates, and pads input text for the sentiment model.

    Args:
        text (str): The raw input sentence.
        tokenizer (spm.SentencePieceProcessor): The SentencePiece tokenizer instance.
        text_max_len (int): The maximum sequence length the model expects.

    Returns:
        np.ndarray: A NumPy array of shape (1, text_max_len) ready for model input.
    """
    tokenized_ids = tokenizer.encode_as_ids(text)
    
    # Use Keras's pad_sequences for robust padding and truncation.
    # 'pre' padding and 'post' truncating match the original model's behavior.
    padded_sequence = keras.preprocessing.sequence.pad_sequences(
        [tokenized_ids], maxlen=text_max_len, padding='pre', truncating='post'
    )
    return padded_sequence.astype(np.int32)