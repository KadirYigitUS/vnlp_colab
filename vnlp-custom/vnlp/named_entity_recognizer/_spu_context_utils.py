import tensorflow as tf
import numpy as np

from ..utils import create_rnn_stacks, process_word_context

# Constants
TOKEN_PIECE_MAX_LEN = 8
SENTENCE_MAX_LEN = 40

def create_spucontext_ner_model(
    TOKEN_PIECE_MAX_LEN,
    SENTENCE_MAX_LEN,
    VOCAB_SIZE,
    ENTITY_VOCAB_SIZE,
    WORD_EMBEDDING_DIM,
    WORD_EMBEDDING_MATRIX,
    NUM_RNN_UNITS,
    NUM_RNN_STACKS,
    FC_UNITS_MULTIPLIER,
    DROPOUT
):
    """
    Builds the SPUContext NER model using the Keras Functional API.

    This refactored version is compatible with modern Keras, more efficient,
    and easier to understand and maintain.
    """
    # 1. Define Explicit Functional API Inputs
    word_input = tf.keras.layers.Input(shape=(TOKEN_PIECE_MAX_LEN,), dtype=tf.int32, name='word_input')
    left_context_input = tf.keras.layers.Input(shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), dtype=tf.int32, name='left_context_input')
    right_context_input = tf.keras.layers.Input(shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN), dtype=tf.int32, name='right_context_input')
    lc_entity_input = tf.keras.layers.Input(shape=(SENTENCE_MAX_LEN, ENTITY_VOCAB_SIZE + 1), dtype=tf.float32, name='lc_entity_input')

    # 2. Define Reusable Sub-models
    word_rnn = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(TOKEN_PIECE_MAX_LEN,)),  # Bug fixed
        tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE, output_dim=WORD_EMBEDDING_DIM,
            embeddings_initializer=tf.keras.initializers.Constant(WORD_EMBEDDING_MATRIX),
            trainable=False, name='WORD_EMBEDDING'
        ),
        create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT)
    ], name='WORD_RNN')

    left_context_rnn = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN)),
        tf.keras.layers.TimeDistributed(word_rnn),
        create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT)
    ], name='LEFT_CONTEXT_RNN')

    right_context_rnn = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN)),
        tf.keras.layers.TimeDistributed(word_rnn),
        create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT, GO_BACKWARDS=True)
    ], name='RIGHT_CONTEXT_RNN')

    lc_entity_rnn = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(SENTENCE_MAX_LEN, ENTITY_VOCAB_SIZE + 1)),
        create_rnn_stacks(NUM_RNN_STACKS, NUM_RNN_UNITS, DROPOUT)
    ], name='PREV_ENTITY_RNN')

    # 3. Build the Graph by Calling Sub-models on Inputs
    word_output = word_rnn(word_input)
    left_context_output = left_context_rnn(left_context_input)
    right_context_output = right_context_rnn(right_context_input)
    lc_entity_output = lc_entity_rnn(lc_entity_input)

    # 4. Define the Final Fully Connected Layers
    concatenated_features = tf.keras.layers.Concatenate()([
        word_output, left_context_output, right_context_output, lc_entity_output
    ])
    x = tf.keras.layers.Dense(NUM_RNN_UNITS * FC_UNITS_MULTIPLIER[0], activation='relu')(concatenated_features)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.Dense(NUM_RNN_UNITS * FC_UNITS_MULTIPLIER[1], activation='relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    entity_output = tf.keras.layers.Dense(ENTITY_VOCAB_SIZE + 1, activation='softmax')(x)

    # 5. Create and Return the Final Model
    ner_model = tf.keras.models.Model(
        inputs=[word_input, left_context_input, right_context_input, lc_entity_input],
        outputs=entity_output,
        name='SPUContext_NER_Model'
    )
    return ner_model


def process_single_word_input(w, tokenized_sentence, spu_tokenizer_word, tokenizer_label, int_preds_so_far):
    """
    Prepares input arrays for a single word, correctly encoding the history
    of previously predicted entity tags for the autoregressive loop.
    """
    entity_vocab_size = len(tokenizer_label.word_index) + 1
    current_word_processed, left_context_words_processed, right_context_words_processed = \
        process_word_context(
            w, tokenized_sentence, spu_tokenizer_word,
            SENTENCE_MAX_LEN, TOKEN_PIECE_MAX_LEN
        )

    # Pre-allocate array for previously predicted entity tags
    left_context_preds = np.zeros((SENTENCE_MAX_LEN, entity_vocab_size), dtype=np.float32)

    # Only populate if there are previous predictions
    if w > 0:
        # Create one-hot vectors for all previous predictions
        one_hot_preds = tf.keras.utils.to_categorical(int_preds_so_far, num_classes=entity_vocab_size)
        # Place them in the correct time-steps of the input array
        left_context_preds[SENTENCE_MAX_LEN - w : SENTENCE_MAX_LEN] = one_hot_preds

    # Expand dims for batch_size = 1 and cast to correct dtypes
    return (
        np.expand_dims(current_word_processed, axis=0).astype(np.int32),
        np.expand_dims(left_context_words_processed, axis=0).astype(np.int32),
        np.expand_dims(right_context_words_processed, axis=0).astype(np.int32),
        np.expand_dims(left_context_preds, axis=0) # Already float32
    )