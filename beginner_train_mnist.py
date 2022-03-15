""" From https://www.tensorflow.org/tutorials/quickstart/beginner
"""
import tensorflow as tf


def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = build_model()

    # todo: how does this work
    # predictions = model(x_train[:1]).numpy()
    # tf.nn.softmax(predictions).numpy()
    # todo: assuming the above is figuring out what a digit is, write a function
    # to to this

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)


def build_model():
    model = tf.keras.models.Sequential([
        # todo: what are all these
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    # todo: what's this
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy'])

    return model


if __name__ == "__main__":
    main()
