### **Test Plan: Executing the Conversion in Google Colab**

#### **Cell 1: Installation & Setup**

# @title Cell 1: Install Dependencies
# We use -q to make the output less noisy.
!pip install -q vnlp_colab
!pip install -q "tf2onnx>=1.16.0" "onnxruntime-gpu>=1.18.0"

print("✅ All necessary packages installed successfully.")

#### **Cell 2: Run the Conversion Script**

# @title Cell 2: Convert All Keras Models to ONNX
import logging
import tensorflow as tf
import tf2onnx
from pathlib import Path
from tqdm import tqdm

# --- IMPORTANT: Import directly from the installed vnlp_colab package ---
from vnlp_colab.stemmer.stemmer_colab import StemmerAnalyzer
from vnlp_colab.pos.pos_colab import SPUContextPoS
from vnlp_colab.ner.ner_colab import SPUContextNER, CharNER
from vnlp_colab.dep.dep_colab import SPUContextDP
from vnlp_colab.sentiment.sentiment_colab import SPUCBiGRUSentimentAnalyzer
from vnlp_colab.utils_colab import setup_logging

# --- Define a clear output directory in the Colab environment ---
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
    try:
        model_proto, _ = tf2onnx.convert.from_keras(keras_model, input_signature, opset=OPSET_VERSION)
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
                tf.TensorSpec(shape=(None, 40, 20), dtype=tf.float32, name="lc_pos_input"),
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

    for model_key, config in tqdm(models_to_convert.items(), desc="Converting All Models"):
        for eval_mode in [False, True]:
            mode_suffix = "eval" if eval_mode else "prod"
            full_model_name = f"{model_key}_{mode_suffix}"
            try:
                logger.info(f"\nInstantiating {full_model_name} to load weights...")
                keras_model = config["init"](eval_mode)
                signature = config["signature"]()
                output_path = OUTPUT_DIR / f"{config['filename']}_{mode_suffix}.onnx"
                convert_model(full_model_name, keras_model, signature, output_path)
            except Exception as e:
                logger.error(f"FATAL: Could not process {full_model_name}. Error: {e}")
    logger.info("\n--- All models processed. ---")

# --- Execute the conversion ---
run_all_conversions()

#### **Cell 3: Verification**

# @title Cell 3: Verify Generated Files
#!ls -lh /content/onnx_models/

