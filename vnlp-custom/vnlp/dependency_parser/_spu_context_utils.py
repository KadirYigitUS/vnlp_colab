# File: /home/ben/miniconda3/envs/bookanalysis/lib/python3.12/site-packages/vnlp/dependency_parser/_spu_context_utils.py

import tensorflow as tf
from ..utils import create_rnn_stacks

def create_spucontext_dp_model(
    TOKEN_PIECE_MAX_LEN,
    SENTENCE_MAX_LEN,
    VOCAB_SIZE,
    ARC_LABEL_VECTOR_LEN,
    WORD_EMBEDDING_DIM,
    WORD_EMBEDDING_MATRIX,
    NUM_RNN_UNITS,
    NUM_RNN_STACKS,
    FC_UNITS_MULTIPLIER,
    DROPOUT
):
    """
    Builds a 1:1 replica of the SPUContext DP model architecture.

    This version uses the modern Keras Functional API to precisely match the
    layer structure, parameter counts, and layer sharing of the pre-trained model,
    ensuring maximum accuracy by enabling correct weight loading.
    """
    # ---- 1. Define Explicit Functional API Inputs ----
    word_input = tf.keras.layers.Input(shape=(TOKEN_PIECE_MAX_LEN,), name='word_input', dtype='int32')
    left_context_input = tf.keras.layers.Input(shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), name='left_context_input', dtype='int32')
    right_context_input = tf.keras.layers.Input(shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), name='right_context_input', dtype='int32')
    lc_arc_label_input = tf.keras.layers.Input(shape=(SENTENCE_MAX_LEN, ARC_LABEL_VECTOR_LEN), name='lc_arc_label_input', dtype='float32')

    # ---- 2. Define Reusable Sub-models ----

    # This is the shared block for processing word pieces. It includes the embedding layer
    # and the first stack of recurrent layers, as per the model summary.
    word_rnn_model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(
            VOCAB_SIZE, WORD_EMBEDDING_DIM,
            embeddings_initializer=tf.keras.initializers.Constant(WORD_EMBEDDING_MATRIX),
            trainable=False, name='WORD_EMBEDDING'
        ),
        create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT)
    ], name="WORD_RNN")

    # The Left Context model applies the shared word_rnn_model to each word in the
    # context and then passes the sequence through another stack of GRUs.
    left_context_rnn_model = tf.keras.models.Sequential([
        tf.keras.layers.TimeDistributed(word_rnn_model),
        create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT)
    ], name="LEFT_CONTEXT_RNN")

    # The Right Context model is identical but processes the sequence backwards.
    right_context_rnn_model = tf.keras.models.Sequential([
        tf.keras.layers.TimeDistributed(word_rnn_model),
        create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT, GO_BACKWARDS=True)
    ], name="RIGHT_CONTEXT_RNN")

    # The previous arc-label model processes the history of predictions.
    lc_arc_label_rnn_model = tf.keras.models.Sequential([
        create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT)
    ], name="PREV_ARC_LABEL_RNN")

    # ---- 3. Build the Main Graph by Calling Sub-models on Inputs ----
    word_output = word_rnn_model(word_input)
    left_context_output = left_context_rnn_model(left_context_input)
    right_context_output = right_context_rnn_model(right_context_input)
    lc_arc_label_output = lc_arc_label_rnn_model(lc_arc_label_input)

    # ---- 4. Concatenate Features and Add Final Classification Layers ----
    concatenated_features = tf.keras.layers.Concatenate()([
        word_output, left_context_output, right_context_output, lc_arc_label_output
    ])

    x = tf.keras.layers.Dense(NUM_RNN_UNITS * FC_UNITS_MULTIPLIER[0], activation='relu', name='dense')(concatenated_features)
    x = tf.keras.layers.Dropout(DROPOUT, name='dropout')(x)
    x = tf.keras.layers.Dense(NUM_RNN_UNITS * FC_UNITS_MULTIPLIER[1], activation='relu', name='dense_1')(x)
    x = tf.keras.layers.Dropout(DROPOUT, name='dropout_1')(x)
    arc_label_output = tf.keras.layers.Dense(ARC_LABEL_VECTOR_LEN, activation='sigmoid', name='dense_2')(x)

    # ---- 5. Create and Return the Final Model ----
    final_model = tf.keras.models.Model(
        inputs=[word_input, left_context_input, right_context_input, lc_arc_label_input],
        outputs=arc_label_output,
        name='SPUContext_DP_Model'
    )
    return final_model