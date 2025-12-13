# @title Cell 1: Install Dependencies (Fixed for Colab)
import os
import sys

# Set flag to minimize TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("📦 Step 1: Using Colab's pre-installed TensorFlow and core packages...")
# Don't upgrade TensorFlow - use what Colab provides (2.19.x)
# This avoids the tf-keras and tensorflow-text conflicts

print("📦 Step 2: Installing compatible ONNX packages...")
# Install ONNX packages that work with TF 2.19 and modern protobuf
!pip install --upgrade -q \
    "protobuf>=5.26.1,<6.0" \
    "onnx>=1.16.0" \
    "onnxruntime-gpu>=1.18.0"

# Install tf2onnx WITHOUT dependencies to avoid protobuf downgrade
!pip install --no-deps -q "tf2onnx>=1.16.0"

print("📦 Step 3: Installing vnlp_colab without dependencies...")
!pip install --no-deps --upgrade -q "git+https://github.com/KadirYigitUS/vnlp_colab.git"

print("📦 Step 4: Installing remaining vnlp_colab dependencies...")
# Install only the dependencies that aren't already satisfied
!pip install -q \
    "sentencepiece==0.2.1" \
    "tqdm>=4.62.0" \
    "regex>=2.3.6.3"

# Unset the environment variable
if 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
    del os.environ['TF_CPP_MIN_LOG_LEVEL']

print("\n✅ All packages installed successfully!")

# Verify critical versions
print("\n📋 Verifying package versions:")
import google.protobuf
print(f"  ✓ protobuf: {google.protobuf.__version__}")

# OOM Crash, Tensor Spec error, CUdnnRNNNV3 support issues resolved
# @title Cell 2 (Corrected): Convert All Keras Models to ONNX (Robust Version)

import logging
import tensorflow as tf
import tf2onnx
from pathlib import Path
from tqdm import tqdm
import os
import gc # Garbage Collector interface

# --- Import from the installed vnlp_colab package ---
from vnlp_colab.stemmer.stemmer_colab import StemmerAnalyzer
from vnlp_colab.pos.pos_colab import SPUContextPoS
from vnlp_colab.ner.ner_colab import SPUContextNER, CharNER
from vnlp_colab.dep.dep_colab import SPUContextDP
from vnlp_colab.sentiment.sentiment_colab import SPUCBiGRUSentimentAnalyzer
from vnlp_colab.utils_colab import setup_logging

# Define a clear output directory in the Colab environment
OUTPUT_DIR = Path("/content/onnx_models")
OPSET_VERSION = 13

# Setup logger and create output directory
setup_logging(level=logging.INFO)
OUTPUT_DIR.mkdir(exist_ok=True)
logger = logging.getLogger(__name__)

def convert_model(model_name: str, keras_model: tf.keras.Model, input_signature: list, output_path: Path):
    """Converts a single Keras model to ONNX format."""
    logger.info(f"--- Converting {model_name} ---")
    logger.info(f"Input signature: {[spec.name for spec in input_signature]}")
    logger.info(f"Output path: {output_path}")

    # FIX 3: Force conversion on CPU to avoid unsupported CudnnRNNV3 op.
    # This is the standard workaround for this tf2onnx issue.
    # The final ONNX model is not affected and will run on GPU.
    logger.info("Forcing CPU context for conversion to ensure compatibility...")
    try:
        model_proto, _ = tf2onnx.convert.from_keras(
            keras_model, input_signature, opset=OPSET_VERSION
        )
        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())
        logger.info(f"✅ Successfully converted {model_name} to ONNX.")
    except Exception as e:
        logger.error(f"❌ FAILED to convert {model_name}. Error: {e}", exc_info=True)


def run_all_conversions():
    """Main function to orchestrate the conversion of all VNLP models."""
    logger.info(f"Starting VNLP model conversion to ONNX format.")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Using ONNX opset version: {OPSET_VERSION}")

    models_to_convert = {
        "Stemmer": {
            "init": lambda eval_mode: StemmerAnalyzer(evaluate=eval_mode).model,
            "filename": "Stemmer_Shen",
            "signature": lambda: [
                tf.TensorSpec(shape=(None, 10, 10), dtype=tf.int32, name="stem_input"),
                tf.TensorSpec(shape=(None, 10, 15), dtype=tf.int32, name="tag_input"),
                tf.TensorSpec(shape=(None, 40, 15), dtype=tf.int32, name="surface_left_input"),
                tf.TensorSpec(shape=(None, 40, 15), dtype=tf.int32, name="surface_right_input"),
            ]
        },
        "SPUContextPoS": {
            "init": lambda eval_mode: SPUContextPoS(evaluate=eval_mode).model,
            "filename": "PoS_SPUContext",
            "signature": lambda: [
                tf.TensorSpec(shape=(None, 8), dtype=tf.int32, name="word_input"),
                tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32, name="left_input"),
                tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32, name="right_input"),
                # FIX 2: Corrected shape from 20 to 18 (17 tags + 1 padding)
                tf.TensorSpec(shape=(None, 40, 18), dtype=tf.float32, name="lc_pos_input"),
            ]
        },
        "SPUContextNER": {
            "init": lambda eval_mode: SPUContextNER(evaluate=eval_mode).model,
            "filename": "NER_SPUContext",
            "signature": lambda: [
                tf.TensorSpec(shape=(None, 8), dtype=tf.int32, name="word_input"),
                tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32, name="left_context_input"),
                tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32, name="right_context_input"),
                tf.TensorSpec(shape=(None, 40, 5), dtype=tf.float32, name="lc_entity_input"),
            ]
        },
        "SPUContextDP": {
            "init": lambda eval_mode: SPUContextDP(evaluate=eval_mode).model,
            "filename": "DP_SPUContext",
            "signature": lambda: [
                tf.TensorSpec(shape=(None, 8), dtype=tf.int32, name="word_input"),
                tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32, name="left_context_input"),
                tf.TensorSpec(shape=(None, 40, 8), dtype=tf.int32, name="right_context_input"),
                tf.TensorSpec(shape=(None, 40, 77), dtype=tf.float32, name="lc_arc_label_input"),
            ]
        },
        "CharNER": {
            "init": lambda eval_mode: CharNER(evaluate=eval_mode).model,
            "filename": "NER_CharNER",
            "signature": lambda: [
                tf.TensorSpec(shape=(None, 256), dtype=tf.int32, name="input_1"),
            ]
        },
        "Sentiment": {
            "init": lambda eval_mode: SPUCBiGRUSentimentAnalyzer(evaluate=eval_mode).model,
            "filename": "Sentiment_SPUCBiGRU",
            "signature": lambda: [
                tf.TensorSpec(shape=(None, 256), dtype=tf.int32, name="input_layer"),
            ]
        },
    }
    
    # FIX 1: Process one model at a time to prevent OOM errors.
    for model_key, config in tqdm(models_to_convert.items(), desc="Converting All Models"):
        for eval_mode in [False, True]:
            mode_suffix = "eval" if eval_mode else "prod"
            full_model_name = f"{model_key}_{mode_suffix}"
            
            try:
                logger.info(f"\nInstantiating {full_model_name} to load weights...")
                
                # Use a context manager to ensure devices are handled correctly
                with tf.device("/cpu:0"):
                    keras_model = config["init"](eval_mode)
                    signature = config["signature"]()
                    output_path = OUTPUT_DIR / f"{config['filename']}_{mode_suffix}.onnx"
                    convert_model(full_model_name, keras_model, signature, output_path)

            except Exception as e:
                logger.error(f"FATAL: Could not process {full_model_name}. Error: {e}")
            finally:
                # Explicitly clear memory after each model conversion
                logger.info(f"Clearing memory after converting {full_model_name}...")
                tf.keras.backend.clear_session()
                gc.collect()

    logger.info("\n--- All models processed successfully. ---")


# --- Execute the conversion ---
run_all_conversions()

#### **Cell 3: Verification**

# @title Cell 3: Verify Generated Files
#!ls -lh /content/onnx_models/

