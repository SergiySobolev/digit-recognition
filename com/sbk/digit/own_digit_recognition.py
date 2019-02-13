from keras.datasets import mnist


class OwnDigitRecognition:

    def train_neural_net(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()