from typing import List, Tuple
import pickle
import os

import numpy as np
import tensorflow as tf
import sentencepiece as spm

from ..tokenizer import TreebankWordTokenize
from ..utils import check_and_download, load_keras_tokenizer
# We use the refactored, correct utility file
from ._spu_context_utils import create_spucontext_pos_model, process_single_word_input

# --- Constants and Global Setup (No changes needed here) ---
RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")
PROD_WEIGHTS_LOC = os.path.join(RESOURCES_PATH, "PoS_SPUContext_prod.weights")
EVAL_WEIGHTS_LOC = os.path.join(RESOURCES_PATH, "PoS_SPUContext_eval.weights")
WORD_EMBEDDING_MATRIX_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'resources/SPUTokenized_word_embedding_16k.matrix'))
PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_SPUContext_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/PoS_SPUContext_eval.weights"
WORD_EMBEDDING_MATRIX_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"
SPU_TOKENIZER_WORD_LOC = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'resources/SPU_word_tokenizer_16k.model'))
TOKENIZER_LABEL_LOC = os.path.join(RESOURCES_PATH, "PoS_label_tokenizer.json")
TOKEN_PIECE_MAX_LEN = 8
SENTENCE_MAX_LEN = 40
spu_tokenizer_word = spm.SentencePieceProcessor(model_file=SPU_TOKENIZER_WORD_LOC)
tokenizer_label = load_keras_tokenizer(TOKENIZER_LABEL_LOC)
sp_key_to_index = {spu_tokenizer_word.id_to_piece(id): id for id in range(spu_tokenizer_word.get_piece_size())}
LABEL_VOCAB_SIZE = len(tokenizer_label.word_index)
WORD_EMBEDDING_VOCAB_SIZE = len(sp_key_to_index)
WORD_EMBEDDING_VECTOR_SIZE = 128
NUM_RNN_STACKS = 1
RNN_UNITS_MULTIPLIER = 1
NUM_RNN_UNITS = WORD_EMBEDDING_VECTOR_SIZE * RNN_UNITS_MULTIPLIER
FC_UNITS_MULTIPLIER = (2, 1)
DROPOUT = 0.2
# --- End of Setup ---

class SPUContextPoS:
    """
    SentencePiece Unigram Context Part of Speech Tagger class.

    - This is a context aware Deep GRU based Part of Speech Tagger that uses `SentencePiece Unigram` tokenizer.
    - It achieves 0.9010 Accuracy and 0.7623 F1 macro score on all of test sets of Universal Dependencies 2.9.
    - This implementation is optimized with tf.function for high-performance inference.
    """
    def __init__(self, evaluate: bool):
        """
        Initializes the model. This is a heavyweight operation and should only
        be called once per application session (managed by a Singleton factory).
        """
        # 1. Build the Keras model using the refactored utility.
        self.model = create_spucontext_pos_model(
            TOKEN_PIECE_MAX_LEN, SENTENCE_MAX_LEN, WORD_EMBEDDING_VOCAB_SIZE, LABEL_VOCAB_SIZE,
            WORD_EMBEDDING_VECTOR_SIZE, np.zeros((WORD_EMBEDDING_VOCAB_SIZE, WORD_EMBEDDING_VECTOR_SIZE)),
            NUM_RNN_UNITS, NUM_RNN_STACKS, FC_UNITS_MULTIPLIER, DROPOUT
        )
        
        # 2. Download resources if they don't exist.
        check_and_download(WORD_EMBEDDING_MATRIX_LOC, WORD_EMBEDDING_MATRIX_LINK)
        MODEL_WEIGHTS_LOC = EVAL_WEIGHTS_LOC if evaluate else PROD_WEIGHTS_LOC
        MODEL_WEIGHTS_LINK = EVAL_WEIGHTS_LINK if evaluate else PROD_WEIGHTS_LINK
        check_and_download(MODEL_WEIGHTS_LOC, MODEL_WEIGHTS_LINK)
        
        # 3. Load weights from disk.
        word_embedding_matrix = np.load(WORD_EMBEDDING_MATRIX_LOC)
        with open(MODEL_WEIGHTS_LOC, 'rb') as fp:
            model_weights = pickle.load(fp)
        model_weights.insert(0, word_embedding_matrix)
        
        # 4. Set weights to the model.
        self.model.set_weights(model_weights)
        
        # 5. Compile the prediction step into a static graph for speed.
        self._initialize_compiled_predict_step()

        self.spu_tokenizer_word = spu_tokenizer_word
        self.tokenizer_label = tokenizer_label

    def _initialize_compiled_predict_step(self):
        """
        Creates a compiled TensorFlow function for the model's forward pass.
        This is the core performance optimization.
        """
        # Define the exact shapes and data types the compiled function will expect.
        input_signature = [
            tf.TensorSpec(shape=(1, TOKEN_PIECE_MAX_LEN), dtype=tf.int32, name="word_input"),
            tf.TensorSpec(shape=(1, SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), dtype=tf.int32, name="left_context_input"),
            tf.TensorSpec(shape=(1, SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), dtype=tf.int32, name="right_context_input"),
            tf.TensorSpec(shape=(1, SENTENCE_MAX_LEN, LABEL_VOCAB_SIZE + 1), dtype=tf.float32, name="lc_pos_input"),
        ]

        # Use the @tf.function decorator to trace and compile the model call.
        @tf.function(input_signature=input_signature)
        def predict_step(word_input, left_context_input, right_context_input, lc_pos_input):
            return self.model([word_input, left_context_input, right_context_input, lc_pos_input], training=False)
            
        self.compiled_predict_step = predict_step

    def predict(self, sentence: str) -> List[Tuple[str, str]]:
        """
        Predicts PoS tags for a sentence using an optimized autoregressive loop.

        Args:
            sentence: Input text(sentence).

        Returns:
             List of (token, pos_label).
        """
        tokenized_sentence = TreebankWordTokenize(sentence)
        num_tokens_in_sentence = len(tokenized_sentence)

        # State list to hold integer predictions from previous steps.
        int_preds = []
        
        # This loop is now much faster because the model call inside is compiled.
        for t in range(num_tokens_in_sentence):
            # 1. Prepare inputs using the NumPy helper function.
            X_numpy = process_single_word_input(
                t, tokenized_sentence, self.spu_tokenizer_word, self.tokenizer_label, int_preds
            )
            
            # 2. Convert inputs to Tensors with the correct data types.
            X_tensors = [tf.convert_to_tensor(arr) for arr in X_numpy]

            # 3. Call the FAST, COMPILED prediction function.
            raw_pred = self.compiled_predict_step(*X_tensors).numpy()[0]
            
            # 4. Decode the result and update state for the next iteration.
            int_pred = np.argmax(raw_pred, axis=-1)
            int_preds.append(int_pred)

        # 5. Convert final integer predictions to text labels.
        pos_labels = self.tokenizer_label.sequences_to_texts([[p] for p in int_preds])
        result = list(zip(tokenized_sentence, pos_labels))

        return result