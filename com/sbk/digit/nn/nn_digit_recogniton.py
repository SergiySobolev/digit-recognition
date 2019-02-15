import logging

import imageio
import yaml
import numpy as np


class NNDigitRecognition:

    def __init__(self):
        self.init_logger()
        logging.info("Initializing Digit Recognition Neural Network...")
        self.init_model()
        self.init_randomize_weights_and_biases()
        logging.info("Initializing Digit Recognition Neural Network finished")

    def train(self, inputs, labels):
        logging.info("Starting training network...")

        for epoch in range(self.epoch):
            i = 0
            while i < len(inputs):
                inputs_batch = inputs[i:i + self.batch]
                real_labels = labels[i:i + self.batch]

                # forward pass
                a1, calculated_labels = self.do_forward_propagation(inputs_batch)

                # calculate loss
                loss = self.cross_entropy(calculated_labels, real_labels)
                loss += self.l2_regularization(0.01, self.weight1, self.weight2)
                self.loss.append(loss)

                # backward pass
                delta_y = (calculated_labels - real_labels) / calculated_labels.shape[0]
                delta_hidden_layer = np.dot(delta_y, self.weight2.T)
                delta_hidden_layer[a1 <= 0] = 0  # derivatives of relu

                # backpropagation
                weight2_gradient = np.dot(a1.T, delta_y)  # forward * backward
                bias2_gradient = np.sum(delta_y, axis=0, keepdims=True)

                weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer)
                bias1_gradient = np.sum(delta_hidden_layer, axis=0, keepdims=True)

                # L2 regularization
                weight2_gradient += 0.01 * self.weight2
                weight1_gradient += 0.01 * self.weight1

                # stochastic gradient descent
                self.weight1 -= self.lr * weight1_gradient  # update weight and bias
                self.bias1 -= self.lr * bias1_gradient
                self.weight2 -= self.lr * weight2_gradient
                self.bias2 -= self.lr * bias2_gradient

                logging.info('=== Epoch: {:d}/{:d}\tIteration:{:d}\tLoss: {:.2f} ==='.format(epoch + 1, self.epoch, i + 1, loss))

                i += self.batch

        logging.info("Training finished")

    def do_forward_propagation(self, inputs_batch):
        z1 = np.dot(inputs_batch, self.weight1) + self.bias1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(a1, self.weight2) + self.bias2
        calculated_labels = self.softmax(z2)
        return a1, calculated_labels

    def test(self, inputs, labels):
        input_layer = np.dot(inputs, self.weight1)
        hidden_layer = np.maximum(input_layer + self.bias1, 0)
        scores = np.dot(hidden_layer, self.weight2) + self.bias2
        probs = self.softmax(scores)
        acc = float(np.sum(np.argmax(probs, 1) == labels)) / float(len(labels))
        logging.info('Test accuracy: {:.2f}%'.format(acc*100))

    def l2_regularization(self, la, weight1, weight2):
        weight1_loss = 0.5 * la * np.sum(weight1 * weight1)
        weight2_loss = 0.5 * la * np.sum(weight2 * weight2)
        return weight1_loss + weight2_loss

    def cross_entropy(self, calculated_labels, real_labels):
        indices = np.argmax(real_labels, axis=1).astype(int)
        probability = calculated_labels[np.arange(len(calculated_labels)), indices]  # inputs[0, indices]
        log = np.log(probability)
        loss = -1.0 * np.sum(log) / len(log)
        return loss

    def softmax(self, inputs):
        exp = np.exp(inputs)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def init_model(self):
        nn_config = yaml.load(open('config/prod.yml'))
        self.lr = nn_config["nn"]["lr"]
        logging.info("Learning rate = %s", self.lr)
        self.epoch = nn_config["nn"]["epoch"]
        logging.info("Epoch number = %s", self.epoch)
        self.batch = nn_config["nn"]["batch"]
        logging.info("Batch size = %s", self.batch)
        self.num_nodes = nn_config["nn"]["num_nodes"]
        logging.info("Number of nodes in different layers = %s", self.num_nodes)
        self.input_nodes = nn_config["nn"]["input_nodes"]
        self.hidden_nodes = nn_config["nn"]["hidden_nodes"]
        self.output_nodes = nn_config["nn"]["output_nodes"]

    def init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def init_randomize_weights_and_biases(self):
        self.weight1 = np.random.normal(0, 1, [self.num_nodes[0], self.num_nodes[1]])
        self.bias1 = np.zeros((1, self.num_nodes[1]))
        self.weight2 = np.random.normal(0, 1, [self.num_nodes[1], self.num_nodes[2]])
        self.bias2 = np.zeros((1, self.num_nodes[2]))
        self.loss = []

    def recognize_digit(self, img_path):
        img_arr = self.get_image(img_path)
        input_layer = np.dot(img_arr, self.weight1)
        hidden_layer = np.maximum(input_layer + self.bias1, 0)
        scores = np.dot(hidden_layer, self.weight2) + self.bias2
        probs = self.softmax(scores)
        return probs.argmax()

    def get_image(self, img_path):
        img = imageio.imread(img_path, as_gray=True).ravel()
        img = self.convert_to_negative(img)
        img = self.normalize(img)
        return np.array(img)

    def normalize(self, i):
        return i / 255

    def convert_to_negative(self, img):
        img = np.abs(255 - img)
        return img




