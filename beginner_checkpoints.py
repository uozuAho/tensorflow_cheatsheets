""" Same as beginner_train_mnist.py, this time saving checkpoints
    along the way, then loading from the checkpoints.

    TBH checkpoints are weird. Prefer load/save model?
"""
import os
import tensorflow as tf


CHECKPOINTS_PATH = './checkpoints/ckpt'


def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = build_model()

    if os.path.exists(CHECKPOINTS_PATH + '.index'):
        print("Loading existing model weights")
        model.load_weights(CHECKPOINTS_PATH)
    else:
        print("No saved weights found, training")
        model.fit(x_train, y_train, epochs=5, callbacks=[build_checkpoint_callback()])

    model.evaluate(x_test,  y_test, verbose=2)


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy'])

    return model


def build_checkpoint_callback():
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINTS_PATH,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=False)


if __name__ == "__main__":
    main()
