import logging

import imageio
import mnist
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

        for e in range(self.epoch):
            for i in range(0, len(inputs), self.batch):
                inputs_batch = inputs[i:i + self.batch]
                real_labels = labels[i:i + self.batch]

                cur_hidden_layer_weights, calculated_labels = self.do_forward_propagation(inputs_batch)

                loss = self.calculate_loss(calculated_labels, real_labels)

                delta_hidden_layer, delta_y = self.do_backward_pass(calculated_labels,
                                                                    cur_hidden_layer_weights,
                                                                    real_labels)

                weight1_gradient, weight2_gradient = self.weight_gradients(cur_hidden_layer_weights,
                                                                           delta_hidden_layer,
                                                                           delta_y,
                                                                           inputs_batch)

                bias1_gradient, bias2_gradient = self.bias_gradients(delta_hidden_layer, delta_y)

                self.update_weights_and_biases(bias1_gradient, bias2_gradient, weight1_gradient, weight2_gradient)

                logging.info('Epoch:%s/%s  Iteration:%s  Loss: %0.2f', e + 1, self.epoch, i + 1, loss)

        logging.info("Training finished")

    def update_weights_and_biases(self, bias1_gradient, bias2_gradient, weight1_gradient, weight2_gradient):
        self.weight1 -= self.lr * weight1_gradient
        self.bias1 -= self.lr * bias1_gradient
        self.weight2 -= self.lr * weight2_gradient
        self.bias2 -= self.lr * bias2_gradient

    def bias_gradients(self, delta_hidden_layer, delta_y):
        bias2_gradient = np.sum(delta_y, axis=0, keepdims=True)
        bias1_gradient = np.sum(delta_hidden_layer, axis=0, keepdims=True)
        return bias1_gradient, bias2_gradient

    def weight_gradients(self, cur_hidden_layer_weights, delta_hidden_layer, delta_y, inputs_batch):
        weight2_gradient = np.dot(cur_hidden_layer_weights.T, delta_y)  # forward * backward
        weight1_gradient = np.dot(inputs_batch.T, delta_hidden_layer)
        weight2_gradient += 0.01 * self.weight2
        weight1_gradient += 0.01 * self.weight1
        return weight1_gradient, weight2_gradient

    def do_backward_pass(self, calculated_labels, cur_hidden_layer_weights, real_labels):
        delta_y = (calculated_labels - real_labels) / calculated_labels.shape[0]
        delta_hidden_layer = np.dot(delta_y, self.weight2.T)
        delta_hidden_layer[cur_hidden_layer_weights <= 0] = 0
        return delta_hidden_layer, delta_y

    def calculate_loss(self, calculated_labels, real_labels):
        loss = self.cross_entropy(calculated_labels, real_labels)
        loss += self.regularization_penalty(0.01, self.weight1, self.weight2)
        self.loss.append(loss)
        return loss

    def do_forward_propagation(self, inputs_batch):
        z1 = np.dot(inputs_batch, self.weight1) + self.bias1
        hidden_layer_weights = np.maximum(z1, 0)
        z2 = np.dot(hidden_layer_weights, self.weight2) + self.bias2
        calculated_labels = self.softmax(z2)
        return hidden_layer_weights, calculated_labels

    def test(self, inputs, labels):
        input_layer = np.dot(inputs, self.weight1)
        hidden_layer = np.maximum(input_layer + self.bias1, 0)
        scores = np.dot(hidden_layer, self.weight2) + self.bias2
        probs = self.softmax(scores)
        acc = float(np.sum(np.argmax(probs, 1) == labels)) / float(len(labels))
        logging.info('Test accuracy: {:.2f}%'.format(acc*100))

    def regularization_penalty(self, la, weight1, weight2):
        weight1_loss = 0.5 * la * np.sum(weight1 ** 2)
        weight2_loss = 0.5 * la * np.sum(weight2 ** 2)
        return weight1_loss + weight2_loss

    def cross_entropy(self, calculated_labels, real_labels):
        indices = np.argmax(real_labels, axis=1).astype(int)
        probability = calculated_labels[np.arange(len(calculated_labels)), indices]
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
        self.input_nodes_num = nn_config["nn"]["input_nodes_num"]
        self.hidden_nodes_num = nn_config["nn"]["hidden_nodes_num"]
        self.output_nodes_num = nn_config["nn"]["output_nodes_num"]
        logging.info("Input layer nodes num = %s", self.input_nodes_num)
        logging.info("Hidden layer nodes num = %s", self.hidden_nodes_num)
        logging.info("Output layer nodes num = %s", self.output_nodes_num)

    def init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def init_randomize_weights_and_biases(self):
        self.weight1 = np.random.normal(0, 1, [self.input_nodes_num, self.hidden_nodes_num])
        self.bias1 = np.zeros((1, self.hidden_nodes_num))
        self.weight2 = np.random.normal(0, 1, [self.hidden_nodes_num, self.output_nodes_num])
        self.bias2 = np.zeros((1, self.output_nodes_num))
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

    def train_on_default_data(self):
        num_classes = 10
        train_images = mnist.train_images()
        train_labels = mnist.train_labels()

        X_train = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2]).astype(
            'float32')
        x_train = X_train / 255
        y_train = np.eye(num_classes)[train_labels]

        self.train(x_train, y_train)



