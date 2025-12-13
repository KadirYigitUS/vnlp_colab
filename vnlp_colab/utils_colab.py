# vnlp_colab/utils_colab.py
# coding=utf-8
# Copyright 2025 VNLP Project Authors.
# Licensed under AGPL-3.0

"""
Core utilities for the VNLP package, refactored for Google Colab.

This module provides essential functions for resource management (downloading,
caching, local package file access), environment setup, and modern Keras 3 /
TensorFlow compatibility.

- Version: 2.2.0
- Keras: 3.x
- TensorFlow: 2.15+
- Python: 3.10+
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Robust package data access
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Fallback for older Python versions if needed
    import importlib_resources as pkg_resources

import numpy as np
import requests
import tensorflow as tf
from tqdm.notebook import tqdm

# --- Environment and Logging Setup ---

__version__ = "2.2.0"
logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures structured logging for the VNLP package.
    Suppresses verbose TensorFlow startup logs.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Suppress verbose TensorFlow warnings for a cleaner output
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    tf.get_logger().setLevel('ERROR')
    logger.info("VNLP logging configured.")
    logger.info(f"Running on TensorFlow v{tf.__version__} and Keras v{tf.keras.__version__}")

def get_vnlp_cache_dir() -> Path:
    """
    Returns the Path object for the VNLP cache directory.
    Defaults to /content/vnlp_cache in a Colab environment.
    """
    cache_dir = Path("/content/vnlp_cache")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create cache directory at {cache_dir}: {e}")
        raise
    return cache_dir

def detect_hardware_strategy() -> tf.distribute.Strategy:
    """
    Detects and returns the appropriate TensorFlow distribution strategy.
    Prioritizes TPU -> Multi-GPU -> Single-GPU -> CPU.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        logger.info(f'Running on TPU {tpu.master()}')
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.TPUStrategy(tpu)
    except (ValueError, tf.errors.NotFoundError):
        logger.debug("TPU not found. Checking for GPUs.")

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.warning("No GPUs found. Using CPU strategy. Inference may be slow.")
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")

    if len(gpus) > 1:
        logger.info(f"Multiple GPUs found ({len(gpus)}). Using MirroredStrategy.")
        return tf.distribute.MirroredStrategy()

    logger.info("Single GPU found. Using OneDeviceStrategy.")
    # Attempt to enable memory growth to prevent allocation errors
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"Could not set memory growth for GPU {gpu.name}: {e}")
            
    return tf.distribute.OneDeviceStrategy(device="/gpu:0")

# --- Resource Management ---

def get_resource_path(package_path: str, resource_name: str) -> Path:
    """
    Robustly gets the path to a resource file within the installed package.
    
    Args:
        package_path: Dot-separated python path (e.g., 'vnlp_colab.resources')
        resource_name: Filename (e.g., 'tokenizer.json')
    """
    try:
        with pkg_resources.path(package_path, resource_name) as path:
            return path
    except (FileNotFoundError, ModuleNotFoundError, ImportError) as e:
        logger.error(f"Resource '{resource_name}' not found in package '{package_path}'. Error: {e}")
        raise

def download_resource(
    file_name: str,
    file_url: str,
    cache_dir: Union[str, Path, None] = None,
    overwrite: bool = False
) -> Path:
    """
    Downloads a file with a progress bar if it doesn't exist in the cache.
    """
    if cache_dir is None:
        cache_dir = get_vnlp_cache_dir()
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    file_path = cache_dir / file_name

    if file_path.exists() and not overwrite:
        logger.info(f"'{file_name}' found in cache. Skipping download.")
        return file_path

    logger.info(f"Downloading '{file_name}' from '{file_url}'...")
    try:
        with requests.get(file_url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            with open(file_path, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {file_name}",
                disable=total_size == 0
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)

            if total_size > 0 and progress_bar.n != total_size:
                logger.warning(f"Download of '{file_name}' might be incomplete.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download '{file_name}': {e}")
        if file_path.exists():
            file_path.unlink()  # Remove partial file
        raise

    logger.info(f"Download of '{file_name}' completed successfully.")
    return file_path

def load_keras_tokenizer(tokenizer_json_path: Union[str, Path]) -> tf.keras.preprocessing.text.Tokenizer:
    """Loads a Keras tokenizer from a JSON file path."""
    logger.info(f"Loading Keras tokenizer from: {tokenizer_json_path}")
    try:
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_json_path}: {e}")
        raise

# --- Keras 3 & TensorFlow Model Utilities ---

def create_rnn_stacks(
    num_rnn_stacks: int,
    num_rnn_units: int,
    dropout: float,
    go_backwards: bool = False
) -> tf.keras.Model:
    """
    Creates a stack of GRU layers wrapped in a Sequential model.
    Compatible with Keras 3 Functional API.
    """
    if num_rnn_stacks < 1:
        raise ValueError("num_rnn_stacks must be at least 1.")

    rnn_layers = []
    # All layers except the last must return sequences
    for _ in range(num_rnn_stacks - 1):
        rnn_layers.append(
            tf.keras.layers.GRU(
                num_rnn_units,
                dropout=dropout,
                return_sequences=True,
                go_backwards=go_backwards
            )
        )
    # The last layer returns the sequence/state as needed by the caller context.
    # Note: For the DP/NER architectures, intermediate stacks often usually need to return sequences 
    # to maintain the 40-step context.
    # However, if this stack serves as a final encoder, it might return a vector.
    # Based on architecture blueprints, the intermediate RNNs usually return sequences (TimeDistributed context).
    # We default to returning sequences False for the final layer here to match 'encoder' behavior, 
    # BUT specific architectures usually override or use this differently.
    # Correction: The original architecture often has stacks where the last GRU returns a single vector 
    # OR returns sequences depending on if it's wrapped in TimeDistributed.
    # For safety in Keras 3, we define the stack.
    
    rnn_layers.append(
        tf.keras.layers.GRU(
            num_rnn_units,
            dropout=dropout,
            return_sequences=False, # Default to encoder behavior
            go_backwards=go_backwards
        )
    )
    return tf.keras.Sequential(rnn_layers)

def tokenize_single_word(
    word: str,
    tokenizer_word: 'spm.SentencePieceProcessor',
    token_piece_max_len: int
) -> np.ndarray:
    """Tokenizes and pads a single word using SentencePiece."""
    tokenized_ids = tokenizer_word.encode_as_ids(word)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        [tokenized_ids],
        maxlen=token_piece_max_len,
        padding='pre',
        truncating='pre'
    )
    return padded[0]

def process_word_context(
    word_index: int,
    sentence_tokens: List[str],
    tokenizer_word: 'spm.SentencePieceProcessor',
    sentence_max_len: int,
    token_piece_max_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized context processor using NumPy pre-allocation.
    Returns: (current_word, left_context, right_context)
    """
    # Create NumPy arrays directly
    current_word = sentence_tokens[word_index]
    left_context = sentence_tokens[:word_index]
    right_context = sentence_tokens[word_index + 1:]

    # 1. Process Current Word
    current_word_processed = tokenize_single_word(
        current_word, tokenizer_word, token_piece_max_len
    ).astype(np.int32)

    # 2. Process Left Context
    # Pre-allocate full buffer with padding (zeros)
    left_context_processed = np.zeros((sentence_max_len, token_piece_max_len), dtype=np.int32)
    
    if left_context:
        # Tokenize all left words
        raw_left = [tokenize_single_word(w, tokenizer_word, token_piece_max_len) for w in left_context]
        raw_left_np = np.array(raw_left, dtype=np.int32)
        
        # Calculate insertion logic
        num_avail = len(raw_left_np)
        if num_avail >= sentence_max_len:
            # Take the last 'sentence_max_len' tokens (closest to current word)
            left_context_processed = raw_left_np[-sentence_max_len:]
        else:
            # Fill from the bottom up (end of the array matches word position)
            left_context_processed[-num_avail:] = raw_left_np

    # 3. Process Right Context
    # Pre-allocate full buffer with padding (zeros)
    right_context_processed = np.zeros((sentence_max_len, token_piece_max_len), dtype=np.int32)
    
    if right_context:
        raw_right = [tokenize_single_word(w, tokenizer_word, token_piece_max_len) for w in right_context]
        raw_right_np = np.array(raw_right, dtype=np.int32)
        
        num_avail = len(raw_right_np)
        if num_avail >= sentence_max_len:
            # Take the first 'sentence_max_len' tokens
            right_context_processed = raw_right_np[:sentence_max_len]
        else:
            # Fill from the top (start of array matches word position)
            right_context_processed[:num_avail] = raw_right_np

    return current_word_processed, left_context_processed, right_context_processed

# --- Main Entry Point for Standalone Use ---

def main() -> None:
    """
    Main function to demonstrate and test the utility functions.
    This can be executed as a standalone script.
    """
    setup_logging()
    logger.info("--- VNLP Colab Utilities Test Suite ---")

    # 1. Test Hardware Detection
    logger.info("\n1. Testing Hardware Detection:")
    strategy = detect_hardware_strategy()
    logger.info(f"   Detected Strategy: {strategy.__class__.__name__}")

    # 2. Test Caching and Downloading
    logger.info("\n2. Testing Caching and Downloading:")
    test_file_url = "https://raw.githubusercontent.com/vngrs-ai/vnlp/main/LICENSE"
    test_file_name = "LICENSE_test.txt"
    try:
        # First download
        license_path = download_resource(test_file_name, test_file_url)
        logger.info(f"   Successfully downloaded to: {license_path}")
        # Second (cached) download
        license_path_cached = download_resource(test_file_name, test_file_url)
        logger.info(f"   Cached path: {license_path_cached}")
    except Exception as e:
        logger.error(f"   Download test failed: {e}")
    
    # 3. Test Package Resource Loading
    logger.info("\n3. Testing Package Resource Loading:")
    try:
        # This will only work if the package is installed
        path = get_resource_path('vnlp_colab.stemmer.resources', 'ExactLookup.txt')
        logger.info(f"   Successfully found package resource: {path}")
        assert path.exists()
    except Exception as e:
        logger.warning(f"   Could not find package resource. This is expected if package is not installed. Error: {e}")


if __name__ == "__main__":
    main()