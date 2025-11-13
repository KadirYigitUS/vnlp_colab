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
Core utilities for the VNLP package, refactored for Google Colab.

This module provides essential functions for resource management (downloading,
caching), environment setup (hardware detection, logging), and modern Keras 3 /
TensorFlow compatibility. It is designed to be the backbone of the vnlp_colab
package, ensuring efficient and maintainable execution.

- Version: 1.0.0
- Keras: 3.x
- TensorFlow: 2.16+
- Python: 3.10+
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import requests
import tensorflow as tf
from tqdm.notebook import tqdm

# --- Environment and Logging Setup ---

__version__ = "1.0.0"
logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO) -> None:
    """Configures structured logging for the VNLP package."""
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

    Defaults to /content/vnlp_cache in a Colab-like environment.
    Creates the directory if it does not exist.

    Returns:
        Path: The path to the cache directory.
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

    - TPUStrategy for TPUs.
    - MirroredStrategy for multiple GPUs.
    - OneDeviceStrategy for a single GPU or CPU.

    Returns:
        tf.distribute.Strategy: The detected distribution strategy.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        logger.info(f'Running on TPU {tpu.master()}')
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        return tf.distribute.TPUStrategy(tpu)
    except (ValueError, tf.errors.NotFoundError):
        logger.info("TPU not found. Checking for GPUs.")

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.info("No GPUs found. Using CPU strategy.")
        return tf.distribute.OneDeviceStrategy(device="/cpu:0")

    if len(gpus) > 1:
        logger.info(f"Multiple GPUs found ({len(gpus)}). Using MirroredStrategy.")
        return tf.distribute.MirroredStrategy()

    logger.info("Single GPU found. Using OneDeviceStrategy.")
    # Set memory growth to avoid allocating all GPU memory at once
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"Could not set memory growth for GPU {gpu.name}: {e}")
    return tf.distribute.OneDeviceStrategy(device="/gpu:0")

# --- Resource Management ---

def download_resource(
    file_name: str,
    file_url: str,
    cache_dir: Union[str, Path, None] = None,
    overwrite: bool = False
) -> Path:
    """
    Checks if a file exists and downloads it if not, with a progress bar.

    Args:
        file_name (str): The name of the file to save.
        file_url (str): The URL to download the file from.
        cache_dir (Union[str, Path, None], optional): Directory to cache the file.
            Defaults to the standard VNLP cache.
        overwrite (bool, optional): If True, re-downloads the file even if it exists.
            Defaults to False.

    Returns:
        Path: The local path to the downloaded file.

    Raises:
        requests.exceptions.RequestException: If the download fails.
    """
    if cache_dir is None:
        cache_dir = get_vnlp_cache_dir()
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    file_path = cache_dir / file_name

    if file_path.exists() and not overwrite:
        logger.info(f"'{file_name}' already exists at '{file_path}'. Skipping download.")
        return file_path

    logger.info(f"Downloading '{file_name}' from '{file_url}'...")
    try:
        with requests.get(file_url, stream=True) as response:
            response.raise_for_status()
            # Use .get('content-length', '0') to safely handle missing header
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB

            with open(file_path, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {file_name}",
                disable=total_size == 0 # Disable bar if total size is unknown
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)

            # --- FIX: Only warn if content-length was provided and doesn't match ---
            if total_size > 0 and progress_bar.n != total_size:
                logger.warning(f"Download of '{file_name}' might be incomplete. "
                               f"Expected {total_size} bytes, got {progress_bar.n} bytes.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download '{file_name}': {e}")
        # Clean up partial download
        if file_path.exists():
            file_path.unlink()
        raise

    logger.info(f"Download of '{file_name}' completed successfully to '{file_path}'.")
    return file_path

def load_keras_tokenizer(tokenizer_json_path: Union[str, Path]) -> tf.keras.preprocessing.text.Tokenizer:
    """
    Loads a Keras tokenizer from a JSON file.

    Args:
        tokenizer_json_path (Union[str, Path]): Path to the tokenizer JSON file.

    Returns:
        tf.keras.preprocessing.text.Tokenizer: The loaded tokenizer object.
    """
    logger.info(f"Loading Keras tokenizer from: {tokenizer_json_path}")
    try:
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    except FileNotFoundError:
        logger.error(f"Tokenizer file not found at: {tokenizer_json_path}")
        raise
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
    Creates a stack of GRU layers with dropout, compatible with Keras 3.

    Args:
        num_rnn_stacks (int): The total number of GRU layers in the stack.
        num_rnn_units (int): The number of units in each GRU layer.
        dropout (float): The dropout rate to apply between layers.
        go_backwards (bool, optional): If True, process sequences in reverse.
            Defaults to False.

    Returns:
        tf.keras.Model: A Sequential model containing the stack of GRU layers.
    """
    if num_rnn_stacks < 1:
        raise ValueError("num_rnn_stacks must be at least 1.")

    rnn_layers = []
    # All but the last layer should return sequences
    for _ in range(num_rnn_stacks - 1):
        rnn_layers.append(
            tf.keras.layers.GRU(
                num_rnn_units,
                dropout=dropout,
                return_sequences=True,
                go_backwards=go_backwards
            )
        )
    # The last layer returns only the final state
    rnn_layers.append(
        tf.keras.layers.GRU(
            num_rnn_units,
            dropout=dropout,
            return_sequences=False,
            go_backwards=go_backwards
        )
    )
    return tf.keras.Sequential(rnn_layers)


def tokenize_single_word(
    word: str,
    tokenizer_word: 'spm.SentencePieceProcessor',
    token_piece_max_len: int
) -> np.ndarray:
    """
    Tokenizes and pads a single word using a SentencePiece tokenizer.

    Args:
        word (str): The input word.
        tokenizer_word (spm.SentencePieceProcessor): The SentencePiece tokenizer instance.
        token_piece_max_len (int): The maximum length for padding/truncating.

    Returns:
        np.ndarray: A 1D NumPy array of token IDs.
    """
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
    Processes the context (left, current, right) for a single word in a sentence.

    Args:
        word_index (int): Index of the current word in the sentence.
        sentence_tokens (List[str]): The list of all tokens in the sentence.
        tokenizer_word: The SentencePiece tokenizer instance.
        sentence_max_len (int): The max length of the context window on each side.
        token_piece_max_len (int): The max length of token pieces for a single word.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the processed
            current word, left context, and right context as NumPy arrays.
    """
    current_word = sentence_tokens[word_index]
    left_context = sentence_tokens[:word_index]
    right_context = sentence_tokens[word_index + 1:]

    # Process current word
    current_word_processed = tokenize_single_word(
        current_word, tokenizer_word, token_piece_max_len
    ).astype(np.int32)

    # Process left context
    left_context_processed = np.array([
        tokenize_single_word(w, tokenizer_word, token_piece_max_len) for w in left_context
    ], dtype=np.int32)
    
    # Pre-pad and truncate left context
    num_left_pad = sentence_max_len - len(left_context_processed)
    if num_left_pad > 0:
        padding = np.zeros((num_left_pad, token_piece_max_len), dtype=np.int32)
        left_context_processed = np.vstack((padding, left_context_processed))
    elif num_left_pad < 0:
        left_context_processed = left_context_processed[-sentence_max_len:]

    # Process right context
    right_context_processed = np.array([
        tokenize_single_word(w, tokenizer_word, token_piece_max_len) for w in right_context
    ], dtype=np.int32)

    # Post-pad and truncate right context
    num_right_pad = sentence_max_len - len(right_context_processed)
    if num_right_pad > 0:
        padding = np.zeros((num_right_pad, token_piece_max_len), dtype=np.int32)
        right_context_processed = np.vstack((right_context_processed, padding))
    elif num_right_pad < 0:
        right_context_processed = right_context_processed[:sentence_max_len]

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

    # 3. Test Usage Snippet
    logger.info("\n3. Example Usage Snippet:")
    print("\nfrom utils_colab import setup_logging, download_resource, detect_hardware_strategy")
    print("setup_logging()")
    print("strategy = detect_hardware_strategy()")
    print("# model = load_model(strategy) # In your model loading script")
    print("# print(preprocess_text('Sample input for VNLP'))")

if __name__ == "__main__":
    main()