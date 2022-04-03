""" Same as beginner_train_mnist.py, this time saving and loading
    the model. Also see beginner_checkpoints.py
"""
import os
import tensorflow as tf
from keras.models import load_model


def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if os.path.exists('model.h5'):
        print('Using saved model')
        model = load_model('model.h5')
    else:
        print('Training model')
        model = build_model()
        model.fit(x_train, y_train, epochs=5)
        model.save('model.h5')

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


if __name__ == "__main__":
    main()
