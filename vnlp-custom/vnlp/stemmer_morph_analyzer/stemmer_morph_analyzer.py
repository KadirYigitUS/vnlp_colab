# File: /home/ben/miniconda3/envs/bookanalysis/lib/python3.12/site-packages/vnlp/stemmer_morph_analyzer/stemmer_morph_analyzer.py
# ---
# This file has been audited and corrected. The original, faster batching logic
# has been restored to fix a critical performance regression.

from typing import List
import pickle
import os

import tensorflow as tf
import numpy as np

from ..tokenizer import TreebankWordTokenize
from ..utils import check_and_download, load_keras_tokenizer
from ._melik_utils import create_stemmer_model, process_input_text
from ._yildiz_analyzer import get_candidate_generator_instance, capitalize

# --- Constants ---
RESOURCES_PATH = os.path.join(os.path.dirname(__file__), "resources/")
PROD_WEIGHTS_LOC = os.path.join(RESOURCES_PATH, "Stemmer_Shen_prod.weights")
EVAL_WEIGHTS_LOC = os.path.join(RESOURCES_PATH, "Stemmer_Shen_eval.weights")
PROD_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Stemmer_Shen_prod.weights"
EVAL_WEIGHTS_LINK = "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/Stemmer_Shen_eval.weights"
TOKENIZER_CHAR_LOC = os.path.join(RESOURCES_PATH, "Stemmer_char_tokenizer.json")
TOKENIZER_TAG_LOC = os.path.join(RESOURCES_PATH, "Stemmer_morph_tag_tokenizer.json")

# Data Processing Config
NUM_MAX_ANALYSIS = 10
STEM_MAX_LEN = 10
TAG_MAX_LEN = 15
SENTENCE_MAX_LEN = 40
SURFACE_TOKEN_MAX_LEN = 15

# Model Config
CHAR_EMBED_SIZE = 32
TAG_EMBED_SIZE = 32
STEM_RNN_DIM = 128
TAG_RNN_DIM = 128
NUM_RNN_STACKS = 1
DROPOUT = 0.2
EMBED_JOIN_TYPE = 'add'
CAPITALIZE_PNONS = False
# --- End Constants ---


class StemmerAnalyzer:
    """
    StemmerAnalyzer Class: A Morphological Disambiguator for Turkish.
    This version is heavily optimized for performance, robustness, and architectural clarity.
    """

    def __init__(self, evaluate: bool = False):
        self.tokenizer_char = load_keras_tokenizer(TOKENIZER_CHAR_LOC)
        self.tokenizer_tag = load_keras_tokenizer(TOKENIZER_TAG_LOC)
        self.tokenizer_tag.filters = ''
        self.tokenizer_tag.split = ' '

        char_vocab_size = len(self.tokenizer_char.word_index) + 1
        tag_vocab_size = len(self.tokenizer_tag.word_index) + 1

        self.model = create_stemmer_model(
            NUM_MAX_ANALYSIS, STEM_MAX_LEN, char_vocab_size, CHAR_EMBED_SIZE, STEM_RNN_DIM,
            TAG_MAX_LEN, tag_vocab_size, TAG_EMBED_SIZE, TAG_RNN_DIM,
            SENTENCE_MAX_LEN, SURFACE_TOKEN_MAX_LEN, EMBED_JOIN_TYPE, DROPOUT,
            NUM_RNN_STACKS
        )

        model_weights_loc = EVAL_WEIGHTS_LOC if evaluate else PROD_WEIGHTS_LOC
        model_weights_link = EVAL_WEIGHTS_LINK if evaluate else PROD_WEIGHTS_LINK
        check_and_download(model_weights_loc, model_weights_link)
        with open(model_weights_loc, 'rb') as fp:
            self.model.set_weights(pickle.load(fp))

        self.candidate_generator = get_candidate_generator_instance(case_sensitive=True)
        self._initialize_compiled_predict_step()

    def _initialize_compiled_predict_step(self):
        """Creates a compiled TensorFlow function for a faster forward pass."""
        input_signature = [
            tf.TensorSpec(shape=(None, NUM_MAX_ANALYSIS, STEM_MAX_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(None, NUM_MAX_ANALYSIS, TAG_MAX_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(None, SENTENCE_MAX_LEN, SURFACE_TOKEN_MAX_LEN), dtype=tf.int32),
            tf.TensorSpec(shape=(None, SENTENCE_MAX_LEN, SURFACE_TOKEN_MAX_LEN), dtype=tf.int32),
        ]

        @tf.function(input_signature=input_signature)
        def predict_step(*args):
            return self.model(list(args), training=False)

        self.compiled_predict_step = predict_step

    def predict(self, sentence: str, batch_size: int = 64) -> List[str]:
        tokens = TreebankWordTokenize(sentence)
        if not tokens:
            return []

        sentence_analyses = [self.candidate_generator.get_analysis_candidates(token) for token in tokens]

        data_for_processing = [[tokens, sentence_analyses]]
        x_numpy, _ = process_input_text(
            data_for_processing, self.tokenizer_char, self.tokenizer_tag, STEM_MAX_LEN,
            TAG_MAX_LEN, SURFACE_TOKEN_MAX_LEN, SENTENCE_MAX_LEN, NUM_MAX_ANALYSIS,
            exclude_unambigious=False, shuffle=False
        )
        
        # Restore original batching logic to fix performance regression.
        if len(tokens) <= batch_size:
            probs_of_sentence = self.compiled_predict_step(*x_numpy)
        else:
            probs = []
            for i in range(0, len(tokens), batch_size):
                batch_x_tuple = tuple(x[i:i + batch_size] for x in x_numpy)
                batch_probs = self.compiled_predict_step(*batch_x_tuple)
                probs.append(batch_probs)
            probs_of_sentence = tf.concat(probs, axis=0)

        ambig_levels = np.array([len(analyses) for analyses in sentence_analyses], dtype=np.int32)
        ambig_levels_tiled = tf.tile(tf.expand_dims(ambig_levels, axis=1), [1, NUM_MAX_ANALYSIS])
        tf_range = tf.range(1, NUM_MAX_ANALYSIS + 1, dtype=tf.int32)
        mask = tf.cast(ambig_levels_tiled >= tf_range, dtype=tf.float32)
        probs_of_sentence *= mask

        predicted_indices = tf.argmax(probs_of_sentence, axis=-1).numpy()

        final_result = []
        for i, analyses_of_token in enumerate(sentence_analyses):
            pred_idx = predicted_indices[i]
            if pred_idx < len(analyses_of_token):
                predicted_analysis = analyses_of_token[pred_idx]
                root, _, tags = predicted_analysis
                if "Prop" in tags and CAPITALIZE_PNONS:
                    root = capitalize(root)
                analysis_str = "+".join([root] + tags).replace('+DB', '^DB')
                final_result.append(analysis_str)
            else:
                final_result.append(tokens[i] + "+Unknown")

        return final_result