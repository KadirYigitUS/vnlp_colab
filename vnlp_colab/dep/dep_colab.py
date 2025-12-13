# vnlp_colab/dep/dep_colab.py
# coding=utf-8
# Copyright 2025 VNLP Project Authors.
# Licensed under AGPL-3.0

"""
Dependency Parser (DP) module for VNLP Colab.

This module provides the high-level DependencyParser API and implementations
for SPUContextDP and the dependency-aware TreeStackDP.

CRITICAL UPDATE:
- Implements automatic chunking for sentences > 40 tokens to prevent
  TensorFlow graph broadcasting errors.
"""

import logging
import pickle
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import sentencepiece as spm
import tensorflow as tf
from tensorflow import keras

# Updated imports for package structure
from vnlp_colab.utils_colab import download_resource, load_keras_tokenizer, get_vnlp_cache_dir, get_resource_path
from vnlp_colab.dep.dep_utils_colab import (
    create_spucontext_dp_model, process_dp_input,
    decode_arc_label_vector, dp_pos_to_displacy_format
)
from vnlp_colab.dep.dep_treestack_utils_colab import (
    create_treestack_dp_model, process_treestack_dp_input
)
from vnlp_colab.stemmer.stemmer_colab import StemmerAnalyzer, get_stemmer_analyzer
from vnlp_colab.pos.pos_colab import PoSTagger

logger = logging.getLogger(__name__)

# --- Model & Resource Configuration ---
_MODEL_CONFIGS = {
    'SPUContextDP': {
        'weights_prod': ("DP_SPUContext_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_SPUContext_prod.weights"),
        'weights_eval': ("DP_SPUContext_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_SPUContext_eval.weights"),
        'word_embedding_matrix': ("SPUTokenized_word_embedding_16k.matrix", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/SPUTokenized_word_embedding_16k.matrix"),
        'spu_tokenizer': "SPU_word_tokenizer_16k.model",
        'label_tokenizer': "DP_label_tokenizer.json",
        'params': {'sentence_max_len': 40, 'word_embedding_dim': 128, 'num_rnn_stacks': 2, 'rnn_units_multiplier': 2, 'fc_units_multiplier': (2, 1), 'dropout': 0.2}
    },
    'TreeStackDP': {
        'weights_prod': ("DP_TreeStack_prod.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_TreeStack_prod.weights"),
        'weights_eval': ("DP_TreeStack_eval.weights", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/DP_TreeStack_eval.weights"),
        'word_embedding_matrix': ("TBWTokenized_word_embedding.matrix", "https://vnlp-model-weights.s3.eu-west-1.amazonaws.com/TBWTokenized_word_embedding.matrix"),
        'word_tokenizer': "TB_word_tokenizer.json",
        'morph_tag_tokenizer': "Stemmer_morph_tag_tokenizer.json",
        'pos_label_tokenizer': "PoS_label_tokenizer.json",
        'dp_label_tokenizer': "DP_label_tokenizer.json",
        'params': {'word_embedding_vector_size': 128, 'pos_embedding_vector_size': 8, 'num_rnn_stacks': 2, 'tag_num_rnn_units': 128, 'lc_num_rnn_units': 384, 'lc_arc_label_num_rnn_units': 384, 'rc_num_rnn_units': 384, 'fc_units_multipliers': (8, 4), 'word_form': 'whole', 'dropout': 0.2, 'sentence_max_len': 40, 'tag_max_len': 15}
    }
}

# --- Singleton Caching ---
_MODEL_INSTANCE_CACHE: Dict[str, Any] = {}

class SPUContextDP:
    """
    SentencePiece Unigram Context Dependency Parser.
    Optimized with tf.function and chunking for high-performance inference.
    """
    def __init__(self, evaluate: bool = False):
        logger.info(f"Initializing SPUContextDP model (evaluate={evaluate})...")
        config = _MODEL_CONFIGS['SPUContextDP']
        self.params_config = config['params'] 
        cache_dir = get_vnlp_cache_dir()

        spu_tokenizer_path = get_resource_path("vnlp_colab.resources", config['spu_tokenizer'])
        label_tokenizer_path = get_resource_path("vnlp_colab.dep.resources", config['label_tokenizer'])
        
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        embedding_path = download_resource(*config['word_embedding_matrix'], cache_dir)

        self.spu_tokenizer_word = spm.SentencePieceProcessor(model_file=str(spu_tokenizer_path))
        self.tokenizer_label = load_keras_tokenizer(label_tokenizer_path)
        self.label_vocab_size = len(self.tokenizer_label.word_index)
        self.arc_label_vector_len = self.params_config['sentence_max_len'] + 1 + self.label_vocab_size + 1
        word_embedding_matrix = np.load(embedding_path)

        params = self.params_config
        num_rnn_units = params['word_embedding_dim'] * params['rnn_units_multiplier']
        
        self.model = create_spucontext_dp_model(
            vocab_size=self.spu_tokenizer_word.get_piece_size(),
            arc_label_vector_len=self.arc_label_vector_len,
            word_embedding_dim=params['word_embedding_dim'],
            word_embedding_matrix=np.zeros_like(word_embedding_matrix),
            num_rnn_units=num_rnn_units,
            num_rnn_stacks=params['num_rnn_stacks'],
            fc_units_multiplier=params['fc_units_multiplier'],
            dropout=params['dropout']
        )
        with open(weights_path, 'rb') as fp: model_weights = pickle.load(fp)
        self.model.set_weights([word_embedding_matrix] + model_weights)
        self._initialize_compiled_predict_step()
        logger.info("SPUContextDP model initialized successfully.")

    def _initialize_compiled_predict_step(self):
        # Static shape (None, 40, ...) allows TF to optimize the graph aggressively
        input_signature = [
            tf.TensorSpec(shape=(None, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32),
            tf.TensorSpec(shape=(None, 40, self.arc_label_vector_len), dtype=tf.float32),
        ]

        @tf.function(input_signature=input_signature)
        def predict_step(word, left_ctx, right_ctx, lc_arc_label_history):
            return self.model([word, left_ctx, right_ctx, lc_arc_label_history], training=False)
        self.compiled_predict_step = predict_step

    def predict(self, tokens: List[str]) -> List[Tuple[int, str, int, str]]:
        if not tokens:
            return []
        return self.predict_batch([tokens])[0]

    def predict_batch(self, batch_of_tokens: List[List[str]]) -> List[List[Tuple[int, str, int, str]]]:
        if not any(batch_of_tokens):
            return [[] for _ in batch_of_tokens]

        # --- STEP 1: CHUNKING LOGIC ---
        # The model accepts max 40 tokens. We must flatten the batch into chunks <= 40.
        chunked_batch = []
        mapping_info = [] # Stores (original_index, start_offset) for reconstruction

        for i, tokens in enumerate(batch_of_tokens):
            if not tokens:
                continue
            
            # Split tokens into chunks of 40
            for chunk_start in range(0, len(tokens), 40):
                chunk = tokens[chunk_start : chunk_start + 40]
                chunked_batch.append(chunk)
                mapping_info.append((i, chunk_start))

        if not chunked_batch:
            return [[] for _ in batch_of_tokens]

        # --- STEP 2: BATCH INFERENCE ON CHUNKS ---
        chunk_batch_size = len(chunked_batch)
        # We know chunks are <= 40, so max_len is safe
        max_len = max(len(c) for c in chunked_batch)
        
        # Prepare result containers for chunks
        chunk_arcs = [[] for _ in range(chunk_batch_size)]
        chunk_labels = [[] for _ in range(chunk_batch_size)]

        for t in range(max_len):
            active_indices, word_batch, left_ctx_batch, right_ctx_batch, lc_hist_batch = [], [], [], [], []
            
            for i in range(chunk_batch_size):
                if t < len(chunked_batch[i]):
                    active_indices.append(i)
                    inputs_np = process_dp_input(
                        t, chunked_batch[i], self.spu_tokenizer_word, self.tokenizer_label,
                        self.arc_label_vector_len, chunk_arcs[i], chunk_labels[i]
                    )
                    word_batch.append(inputs_np[0])
                    left_ctx_batch.append(inputs_np[1])
                    right_ctx_batch.append(inputs_np[2])
                    lc_hist_batch.append(inputs_np[3])

            if not active_indices:
                break

            inputs_tf = [
                tf.convert_to_tensor(np.vstack(word_batch)),
                tf.convert_to_tensor(np.vstack(left_ctx_batch)),
                tf.convert_to_tensor(np.vstack(right_ctx_batch)),
                tf.convert_to_tensor(np.vstack(lc_hist_batch)),
            ]

            logits_batch = self.compiled_predict_step(*inputs_tf).numpy()
            
            for i, logits in enumerate(logits_batch):
                original_index = active_indices[i]
                arc, label = decode_arc_label_vector(logits, self.params_config['sentence_max_len'], self.label_vocab_size)
                chunk_arcs[original_index].append(arc)
                chunk_labels[original_index].append(label)
        
        # --- STEP 3: RECONSTRUCTION ---
        final_results = [[] for _ in range(len(batch_of_tokens))]
        
        for i, (orig_idx, offset) in enumerate(mapping_info):
            chunk_tokens = chunked_batch[i]
            chunk_arc_ids = chunk_arcs[i]
            chunk_label_ids = chunk_labels[i]
            
            labels_str = self.tokenizer_label.sequences_to_texts([[lbl] for lbl in chunk_label_ids])
            
            chunk_result = []
            for idx, (token, label) in enumerate(zip(chunk_tokens, labels_str)):
                # Adjust Head Index:
                # The model predicts heads relative to the chunk (0..40).
                # We need to map this back to the absolute sentence position if possible.
                # However, dependency parsers usually can't predict heads outside their window.
                # We keep the relative head prediction but note the offset for the token ID.
                # Absolute token ID = offset + idx + 1
                
                # Note on Head Adjustment: 
                # If predicted head is 0 (ROOT), it stays 0.
                # If predicted head is > 0, it means "the Xth token in this chunk".
                # Technically, this head points to local chunk index. 
                # Converting to absolute index requires assuming the head is inside the chunk.
                head_idx = chunk_arc_ids[idx]
                if head_idx > 0:
                    head_idx += offset
                
                chunk_result.append(
                    (offset + idx + 1, token, head_idx, label or "UNK")
                )
            
            final_results[orig_idx].extend(chunk_result)

        return final_results

class TreeStackDP:
    """Implementation for the TreeStack Dependency Parser."""
    def __init__(self, stemmer_analyzer: StemmerAnalyzer, pos_tagger: PoSTagger, evaluate: bool = False):
        logger.info(f"Initializing TreeStackDP model (evaluate={evaluate})...")
        self.stemmer_analyzer = stemmer_analyzer
        self.pos_tagger = pos_tagger
        config = _MODEL_CONFIGS['TreeStackDP']
        self.params = config['params']
        cache_dir = get_vnlp_cache_dir()

        word_tok_path = get_resource_path("vnlp_colab.resources", config['word_tokenizer'])
        morph_tok_path = get_resource_path("vnlp_colab.stemmer.resources", config['morph_tag_tokenizer'])
        pos_tok_path = get_resource_path("vnlp_colab.pos.resources", config['pos_label_tokenizer'])
        dp_tok_path = get_resource_path("vnlp_colab.dep.resources", config['dp_label_tokenizer'])
        
        weights_file, weights_url = config['weights_eval'] if evaluate else config['weights_prod']
        weights_path = download_resource(weights_file, weights_url, cache_dir)
        embedding_path = download_resource(*config['word_embedding_matrix'], cache_dir)

        self.tokenizer_word = load_keras_tokenizer(word_tok_path)
        self.tokenizer_morph_tag = load_keras_tokenizer(morph_tok_path)
        self.tokenizer_pos = load_keras_tokenizer(pos_tok_path)
        self.tokenizer_label = load_keras_tokenizer(dp_tok_path)
        
        word_embedding_matrix = np.load(embedding_path)
        tag_embedding_matrix = self.stemmer_analyzer.model.layers[5].weights[0].numpy()
        self.arc_label_vector_len = self.params['sentence_max_len'] + 1 + len(self.tokenizer_label.word_index) + 1
        
        self.model = create_treestack_dp_model(
            word_embedding_vocab_size=len(self.tokenizer_word.word_index) + 1,
            word_embedding_matrix=np.zeros_like(word_embedding_matrix),
            pos_vocab_size=len(self.tokenizer_pos.word_index),
            arc_label_vector_len=self.arc_label_vector_len,
            tag_embedding_matrix=tag_embedding_matrix,
            **self.params
        )
        with open(weights_path, 'rb') as fp: model_weights = pickle.load(fp)
        self.model.set_weights([model_weights[0], word_embedding_matrix] + model_weights[1:])
        logger.info("TreeStackDP model initialized successfully.")

    def predict(self, tokens: List[str]) -> List[Tuple[int, str, int, str]]:
        if not tokens:
            return []
        return self.predict_batch([tokens])[0]

    def predict_batch(self, batch_of_tokens: List[List[str]]) -> List[List[Tuple[int, str, int, str]]]:
        # Fallback to simple batch loop for now as TreeStack architecture handles context differently
        # and implicit chunking is complex due to dependencies on Stemmer/POS alignment.
        # However, we implement basic truncation protection.
        
        final_results = []
        for tokens in batch_of_tokens:
            # Truncate to max len if necessary to prevent crash
            limit = self.params['sentence_max_len']
            safe_tokens = tokens[:limit] 
            
            # --- Pipeline Dependencies ---
            analyses = self.stemmer_analyzer.predict([safe_tokens])
            pos_res = self.pos_tagger.predict(safe_tokens)
            pos_tags = [tag for _, tag in pos_res]
            
            arcs, labels = [], []
            for t in range(len(safe_tokens)):
                inputs = process_treestack_dp_input(
                    safe_tokens, analyses, pos_tags, arcs, labels,
                    self.tokenizer_word, self.tokenizer_morph_tag, self.tokenizer_pos, self.tokenizer_label,
                    self.params['word_form'], self.params['sentence_max_len'], self.params['tag_max_len'],
                    self.arc_label_vector_len
                )
                # Reshape inputs for model: list of arrays
                model_inputs = [np.expand_dims(inp, axis=0) for inp in inputs] # Add batch dim 1
                # Actually process_treestack_dp_input already adds batch dim, but verify shape
                
                # Fix: process_treestack_dp_input returns tuples of (1, ...) arrays.
                # We need to stack them if we were doing true batching, but here we loop.
                # Just use index 0 of the returned tuple for each input.
                
                # For single instance inference:
                # The model expects a list of inputs.
                # inputs tuple is: (word_t, tags_t, pos_t, lc_words, lc_tags, lc_pos, lc_arc, rc_words, rc_tags, rc_pos)
                
                logits = self.model(list(inputs), training=False).numpy()[0]
                arc, label = decode_arc_label_vector(logits, limit, len(self.tokenizer_label.word_index))
                arcs.append(arc)
                labels.append(label)
                
            labels_str = self.tokenizer_label.sequences_to_texts([[lbl] for lbl in labels])
            result = [(idx + 1, token, arcs[idx], label or "UNK")
                      for idx, (token, label) in enumerate(zip(safe_tokens, labels_str))]
            final_results.append(result)
            
        return final_results

class DependencyParser:
    """Main API class for Dependency Parser implementations."""
    def __init__(self, model: str = 'SPUContextDP', evaluate: bool = False):
        self.available_models = ['SPUContextDP', 'TreeStackDP']
        if model not in self.available_models:
            raise ValueError(f"'{model}' is not a valid model. Try one of {self.available_models}")

        cache_key = f"dp_{model}_{'eval' if evaluate else 'prod'}"
        if cache_key not in _MODEL_INSTANCE_CACHE:
            logger.info(f"Instance for '{cache_key}' not found. Creating new one.")
            if model == 'SPUContextDP':
                instance = SPUContextDP(evaluate)
            elif model == 'TreeStackDP':
                stemmer = get_stemmer_analyzer(evaluate)
                pos_tagger = PoSTagger(model='TreeStackPoS', evaluate=evaluate) 
                instance = TreeStackDP(stemmer, pos_tagger, evaluate)
            _MODEL_INSTANCE_CACHE[cache_key] = instance
        else:
            logger.info(f"Found cached instance for '{cache_key}'.")
        self.instance: Union[SPUContextDP, TreeStackDP] = _MODEL_INSTANCE_CACHE[cache_key]

    def predict(
        self, tokens: List[str], displacy_format: bool = False,
        pos_result: List[Tuple[str, str]] = None
    ) -> Union[List[Tuple[int, str, int, str]], List[Dict]]:
        dp_result = self.instance.predict(tokens)
        if displacy_format:
            return dp_pos_to_displacy_format(dp_result, pos_result)
        return dp_result

    def predict_batch(
        self, batch_of_tokens: List[List[str]]
    ) -> List[List[Tuple[int, str, int, str]]]:
        return self.instance.predict_batch(batch_of_tokens)

if __name__ == "__main__":
    from vnlp_colab.tokenizer_colab import TreebankWordTokenize
    setup_logging()
    parser = DependencyParser(model='SPUContextDP')
    # Test long sentence
    long_sent = ["word"] * 50
    print(parser.predict(long_sent))