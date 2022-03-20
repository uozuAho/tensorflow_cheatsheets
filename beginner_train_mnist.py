""" From https://www.tensorflow.org/tutorials/quickstart/beginner
"""
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(x_train.shape)
    # one 28x28 digit:
    print(x_train[0].shape)
    show_digit(x_train[0])
    # note this shape is different, and what the model wants:
    print(x_train[:1].shape)
    # see https://numpy.org/devdocs/user/basics.indexing.html

    # model = build_model()

    # digit = x_train[1]
    # show_digit(digit)
    # print(predict_digit(digit))

    # predictions = model(x_train[:1]).numpy()
    # tf.nn.softmax(predictions).numpy()
    # todo: assuming the above is figuring out what a digit is, write a function
    # to to this

    # model.fit(x_train, y_train, epochs=5)
    # model.evaluate(x_test,  y_test, verbose=2)

    # print(predict_digit(model, x_train[:1]))


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


def predict_digit(model, input):
    # model(x) runs model.predict on x, returns a tensor
    # numpy() converts the result to a numpy array
    # argmax gets the index of the max value
    # see https://keras.io/api/models/model_training_apis/#predict-method
    return model(input).numpy().argmax()


def show_digit(digit):
    fig = plt.figure()
    plt.imshow(digit, cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
