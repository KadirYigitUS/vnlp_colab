import os
import pickle
from typing import Optional, Tuple

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ..utils import check_and_download

# --- Consolidated Helper Functions ---

def create_spucbigru_sentiment_model(text_max_len: int, vocab_size: int, word_embedding_dim: int,
                                     num_rnn_units: int, num_rnn_stacks: int, dropout_rate: float) -> tf.keras.Model:
    """
    Creates a Bidirectional GRU model that precisely matches the architecture of the saved weights.
    """
    inp_layer = tf.keras.layers.Input(shape=(text_max_len,), name="input_layer")
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=word_embedding_dim, trainable=False, name="word_embedding"
    )(inp_layer)
    model_flow = embedding_layer
    for i in range(num_rnn_stacks):
        model_flow = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(num_rnn_units, dropout=dropout_rate, return_sequences=True),
            name=f"bidirectional_gru_{i+1}"
        )(model_flow)
    model_flow = tf.keras.layers.GlobalAveragePooling1D(name="global_avg_pooling")(model_flow)
    model_flow = tf.keras.layers.Dropout(dropout_rate, name="post_pooling_dropout")(model_flow)
    model_flow = tf.keras.layers.Dense(num_rnn_units // 8, activation='relu', name="intermediate_dense")(model_flow)
    model_flow = tf.keras.layers.Dropout(dropout_rate, name="final_dropout")(model_flow)
    out_layer = tf.keras.layers.Dense(1, activation='sigmoid', name="sentiment_output")(model_flow)
    model = tf.keras.models.Model(inputs=inp_layer, outputs=out_layer)
    return model

def process_text_input(text: str, tokenizer: spm.SentencePieceProcessor, text_max_len: int) -> np.ndarray:
    """
    Tokenizes, truncates, and pads input text to match the model's training configuration.
    """
    integer_tokenized_text = tokenizer.encode_as_ids(text)
    # The padding='pre' and truncating='post' combination is crucial.
    # It preserves the beginning of long texts, which matches the model's learned behavior.
    padded_text = pad_sequences([integer_tokenized_text], maxlen=text_max_len, padding='pre', truncating='post')
    return padded_text.astype(np.int32)

# --- Resource Paths and Model Configuration ---

RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")
PROD_WEIGHTS_LOC = os.path.join(RESOURCES_PATH, "Sentiment_SPUCBiGRU_prod.weights")
EVAL_WEIGHTS_LOC = os.path.join(RESOURCES_PATH, "Sentiment_SPUCBiGRU_eval.weights")
WORD_EMBEDDING_MATRIX_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources/SPUTokenized_word_embedding_16k.matrix'))
PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Sentiment_SPUCBiGRU_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Sentiment_SPUCBiGRU_eval.weights"
WORD_EMBEDDING_MATRIX_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"
SPU_TOKENIZER_WORD_LOC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources/SPU_word_tokenizer_16k.model'))

TEXT_MAX_LEN = 256
WORD_EMBEDDING_VECTOR_SIZE = 128
NUM_RNN_STACKS = 3
NUM_RNN_UNITS = 128
DROPOUT = 0.2

# --- Singleton Caching for Model, Compiled Function, and Tokenizer ---

_CACHED_RESOURCES: dict[bool, Tuple] = {}
_TOKENIZER: Optional[spm.SentencePieceProcessor] = None

def _get_or_load_resources(evaluate: bool) -> tuple[tf.keras.Model, callable, spm.SentencePieceProcessor]:
    """
    Loads and caches the model, a compiled inference function, and the tokenizer.
    """
    global _TOKENIZER, _CACHED_RESOURCES
    if evaluate in _CACHED_RESOURCES:
        return _CACHED_RESOURCES[evaluate]

    if _TOKENIZER is None:
        _TOKENIZER = spm.SentencePieceProcessor(SPU_TOKENIZER_WORD_LOC)

    model_weights_loc = EVAL_WEIGHTS_LOC if evaluate else PROD_WEIGHTS_LOC
    model_weights_link = EVAL_WEIGHTS_LINK if evaluate else PROD_WEIGHTS_LINK
    check_and_download(WORD_EMBEDDING_MATRIX_LOC, WORD_EMBEDDING_MATRIX_LINK)
    check_and_download(model_weights_loc, model_weights_link)

    word_embedding_matrix = np.load(WORD_EMBEDDING_MATRIX_LOC)
    with open(model_weights_loc, 'rb') as fp:
        trainable_weights = pickle.load(fp)

    vocab_size = _TOKENIZER.get_piece_size()
    model = create_spucbigru_sentiment_model(
        TEXT_MAX_LEN, vocab_size, WORD_EMBEDDING_VECTOR_SIZE, NUM_RNN_UNITS, NUM_RNN_STACKS, DROPOUT
    )
    full_weights = [word_embedding_matrix] + trainable_weights
    model.set_weights(full_weights)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, TEXT_MAX_LEN], dtype=tf.int32)])
    def inference_function(input_tensor):
        return model(input_tensor, training=False)

    _CACHED_RESOURCES[evaluate] = (model, inference_function, _TOKENIZER)
    return _CACHED_RESOURCES[evaluate]

# --- Main Implementation Class ---

class SPUCBiGRUSentimentAnalyzer:
    """
    SentencePiece Unigram Context Bidirectional GRU Sentiment Analyzer class.
    This class uses a singleton pattern and robust preprocessing to ensure fast,
    accurate, and stable predictions.
    """
    def __init__(self, evaluate: bool = False):
        self.model, self._compiled_predict, self.spu_tokenizer_word = _get_or_load_resources(evaluate)

    def predict(self, text: str) -> int:
        """
        Args:
            text: Input text.
        Returns:
            Sentiment label (1 for positive, 0 for negative).
        """
        prob = self.predict_proba(text)
        return 1 if prob > 0.5 else 0

    def predict_proba(self, text: str) -> float:
        """
        Args:
            text: Input text.
        Returns:
            Probability that the input text has positive sentiment.
        """
        processed_input = process_text_input(text, self.spu_tokenizer_word, TEXT_MAX_LEN)
        prob = self._compiled_predict(tf.constant(processed_input)).numpy()[0][0]
        return float(prob)