""" From https://www.tensorflow.org/tutorials/quickstart/beginner

    Train a NN to classify digits from the MNIST dataset.
"""
import random
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    print("loading training data...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = build_model()

    print("Using the untrained model to read a few digits")
    for _ in range(3):
        idx = random.randint(0, len(x_train))
        digit = x_train[idx:idx+1]
        print('prediction for shown digit:')
        print(predict_digit(model, digit))
        show_digit(digit)

    print("training the model...")
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test,  y_test, verbose=2)

    print("Using the trained model to read a few digits")
    for _ in range(3):
        idx = random.randint(0, len(x_train))
        digit = x_train[idx:idx+1]
        print('prediction for shown digit:')
        print(predict_digit(model, digit))
        show_digit(digit)


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
    # model(x) runs model.predict on x, returns a tensor (logits?)
    # softmax converts logits to probabilities
    # numpy() converts the result to a numpy array
    # argmax gets the index of the max value
    # see https://keras.io/api/models/model_training_apis/#predict-method
    predictions = model(input).numpy()
    return tf.nn.softmax(predictions).numpy().argmax()


def show_digit(digit):
    """ Expects a digit in the shape fed to the model: (1, 28, 28) """

    # A note about the shape of the MNIST data:
    # The training set (x_train) has a shape (60000, 28, 28):
    # 60k 28x28 pixel digits.
    #
    # Note the shape of x_train[0] and x_train[:1] is different
    # The model wants x_train[:1]
    # print(x_train[0].shape) -> (28, 28)
    # print(x_train[:1].shape) -> (1, 28, 28)
    # see https://numpy.org/devdocs/user/basics.indexing.html

    plt.figure()
    plt.imshow(digit[0], cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()
