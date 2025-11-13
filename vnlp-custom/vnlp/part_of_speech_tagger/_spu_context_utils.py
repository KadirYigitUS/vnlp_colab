import tensorflow as tf
import numpy as np

from ..utils import create_rnn_stacks, process_word_context

# Constants defined for clarity, matching the values used in the main class.
TOKEN_PIECE_MAX_LEN = 8
SENTENCE_MAX_LEN = 40


def create_spucontext_pos_model(
    TOKEN_PIECE_MAX_LEN,
    SENTENCE_MAX_LEN,
    VOCAB_SIZE,
    POS_VOCAB_SIZE,
    WORD_EMBEDDING_DIM,
    WORD_EMBEDDING_MATRIX,
    NUM_RNN_UNITS,
    NUM_RNN_STACKS,
    FC_UNITS_MULTIPLIER,
    DROPOUT
):
    """
    Builds the SPUContext PoS tagging model with a corrected architecture.

    This version uses the Functional API correctly with explicit Input layers
    and preserves the intended layer sharing from the original model, ensuring
    compatibility with the provided weights, including all Dropout layers.
    """
    # ---- 1. Define shared Sequential blocks ----

    # WORD_RNN is the common block for processing token pieces into a word vector.
    # It is reused for the current word, left context, and right context.
    word_rnn = tf.keras.models.Sequential(name='WORD_RNN_block')
    word_rnn.add(
        tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=WORD_EMBEDDING_DIM,
            embeddings_initializer=tf.keras.initializers.Constant(WORD_EMBEDDING_MATRIX),
            trainable=False,
            name='WORD_EMBEDDING',
            input_shape=(TOKEN_PIECE_MAX_LEN,)
        )
    )
    word_rnn.add(create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT))

    # ---- 2. Define Functional API inputs ----
    word_input = tf.keras.layers.Input(shape=(TOKEN_PIECE_MAX_LEN,), name='word_input')
    left_input = tf.keras.layers.Input(shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), name='left_input')
    right_input = tf.keras.layers.Input(shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), name='right_input')
    lc_pos_input = tf.keras.layers.Input(shape=(SENTENCE_MAX_LEN, POS_VOCAB_SIZE + 1), name='lc_pos_input')

    # ---- 3. Build the four main parallel branches ----

    # Path 1: Current word processing
    word_output = word_rnn(word_input)

    # Path 2: Left context processing
    left_context_word_vectors = tf.keras.layers.TimeDistributed(word_rnn)(left_input)
    left_context_output = create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT)(left_context_word_vectors)

    # Path 3: Right context processing
    right_context_word_vectors = tf.keras.layers.TimeDistributed(word_rnn)(right_input)
    right_context_output = create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT, GO_BACKWARDS=True)(right_context_word_vectors)

    # Path 4: Previously predicted (left) PoS tags processing
    lc_pos_output = create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT)(lc_pos_input)

    # ---- 4. Concatenate branches and add final FC layers ----
    concatenated = tf.keras.layers.Concatenate()([word_output, left_context_output, right_context_output, lc_pos_output])

    fc_layer_one = tf.keras.layers.Dense(NUM_RNN_UNITS * FC_UNITS_MULTIPLIER[0], activation='relu')(concatenated)
    fc_layer_one_dropout = tf.keras.layers.Dropout(DROPOUT)(fc_layer_one)
    fc_layer_two = tf.keras.layers.Dense(NUM_RNN_UNITS * FC_UNITS_MULTIPLIER[1], activation='relu')(fc_layer_one_dropout)
    fc_layer_two_dropout = tf.keras.layers.Dropout(DROPOUT)(fc_layer_two)
    pos_output = tf.keras.layers.Dense(POS_VOCAB_SIZE + 1, activation='softmax')(fc_layer_two_dropout)

    # ---- 5. Build and return the final model ----
    pos_model = tf.keras.models.Model(
        inputs=[word_input, left_input, right_input, lc_pos_input],
        outputs=pos_output,
        name='SPUContext_PoS_Model'
    )
    return pos_model


def process_single_word_input(
    w,
    tokenized_sentence,
    spu_tokenizer_word,
    tokenizer_label,
    int_preds_so_far
):
    """
    Prepares input arrays for a single word, ensuring correct data types for tf.function.
    """
    entity_vocab_size = len(tokenizer_label.word_index)
    current_word_processed, left_context_words_processed, right_context_words_processed = \
        process_word_context(
            w, tokenized_sentence, spu_tokenizer_word,
            SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN
        )

    # Build one-hot vectors for previous POS tags, pre-padding with zeros
    left_context_preds = []
    num_predictions_to_pad = SENTENCE_MAX_LEN - w
    for _ in range(num_predictions_to_pad):
        left_context_preds.append(np.zeros(entity_vocab_size + 1))

    for w_idx in range(w):
        one_hot_pred = tf.keras.utils.to_categorical(
            int_preds_so_far[w_idx], num_classes=entity_vocab_size + 1
        )
        left_context_preds.append(one_hot_pred)

    # Truncate if sentence is longer than SENTENCE_MAX_LEN
    left_context_preds = np.array(left_context_preds)[-SENTENCE_MAX_LEN:]

    # Expand dims for batch_size = 1 and cast to correct dtypes for the compiled model
    return (
        np.expand_dims(current_word_processed, axis=0).astype(np.int32),
        np.expand_dims(left_context_words_processed, axis=0).astype(np.int32),
        np.expand_dims(right_context_words_processed, axis=0).astype(np.int32),
        np.expand_dims(left_context_preds, axis=0).astype(np.float32)
    )