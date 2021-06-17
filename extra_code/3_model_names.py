import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.models import load_model

#%matplotlib inline

SCRAPED_WINES_INPUT_PATH = "data/scraped/names_prices_descriptions.pickle"
MODEL_WEIGHTS_PATH = "data/models_weights/model_weights_name.h5"

print(tf.__version__)

# tf.logging.set_verbosity(tf.logging.INFO)


def transform(txt, pad_to=None):
    # drop any non-ascii characters
    output = np.asarray([ord(c) for c in txt if ord(c) < 255], dtype=np.int32)
    if pad_to is not None:
        output = output[:pad_to]
        output = np.concatenate([np.zeros([pad_to - len(txt)], dtype=np.int32), output])
    return output


def training_generator(seq_len=100, batch_size=1024):
    """A generator yields (source, target) arrays for training."""
    wine_data = pd.read_pickle(SCRAPED_WINES_INPUT_PATH)
    wine_data = wine_data["name"]  # Take just the names for modeling
    txt = "\n".join(wine_data)

    # tf.logging.info('Input text [%d] %s', len(txt), txt[:50])
    source = transform(txt)
    while True:
        offsets = np.random.randint(0, len(source) - seq_len, batch_size)

        # Our model uses sparse crossentropy loss, but Keras requires labels
        # to have the same rank as the input logits.  We add an empty final
        # dimension to account for this.
        yield (
            np.stack([source[idx : idx + seq_len] for idx in offsets]),
            np.expand_dims(
                np.stack([source[idx + 1 : idx + seq_len + 1] for idx in offsets]), -1
            ),
        )


EMBEDDING_DIM = 512


def lstm_model(seq_len=100, batch_size=None, stateful=True):
    """Language model: predict the next word given the current word."""
    source = tf.keras.Input(
        name="seed", shape=(seq_len,), batch_size=batch_size, dtype=tf.int32
    )

    embedding = tf.keras.layers.Embedding(input_dim=256, output_dim=EMBEDDING_DIM)(
        source
    )
    lstm_1 = tf.keras.layers.LSTM(
        EMBEDDING_DIM, stateful=stateful, return_sequences=True
    )(embedding)
    lstm_2 = tf.keras.layers.LSTM(
        EMBEDDING_DIM, stateful=stateful, return_sequences=True
    )(lstm_1)
    # drop_1 = tf.keras.layers.Dropout(0.2)
    predicted_char = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(256, activation="softmax")
    )(lstm_2)
    model = tf.keras.Model(inputs=[source], outputs=[predicted_char])
    model.compile(
        optimizer="rmsprop",
        # optimizer=tf.keras.optimizers.RMSprop(lr=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

    return model


tf.keras.backend.clear_session()

training_model = lstm_model(seq_len=100, batch_size=1024, stateful=False)
# training_model.load_weights('model_small_chkpt.h5', by_name=True)

checkpoint = keras.callbacks.ModelCheckpoint(
    "model_names_chkpt.h5",
    monitor="sparse_categorical_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)
early_stopping = keras.callbacks.EarlyStopping(
    monitor="sparse_categorical_accuracy", patience=3, mode="max"
)
callbacks_list = [checkpoint, early_stopping]

print(training_model.summary())

training_model.fit_generator(
    training_generator(seq_len=100, batch_size=1024),
    steps_per_epoch=100,
    epochs=2,
    callbacks=callbacks_list,
)

# training_model.save_weights(MODEL_WEIGHTS_PATH, overwrite=True)

