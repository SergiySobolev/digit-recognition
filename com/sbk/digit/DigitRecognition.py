import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import imageio


class DigitRecognition:

    def __init__(self) :
        self.image_width = 28
        self.image_height = 28
        n_input = self.image_width * self.image_height
        n_hidden1 = 512
        n_hidden2 = 256
        n_hidden3 = 128
        n_output = 10

        self.learning_rate = 1e-4
        self.n_iterations = 1200
        self.batch_size = 2048
        self.dropout = 0.5

        self.X = tf.placeholder("float", [None, n_input])
        self.Y = tf.placeholder("float", [None, n_output])
        self.keep_prob = tf.placeholder(tf.float32)

        self.weights = {
            'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
            'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
            'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
            'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
        }

        self.biases = {
            'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
            'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
            'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
            'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
        }

        layer_1 = tf.add(tf.matmul(self.X, self.weights['w1']), self.biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['w2']), self.biases['b2'])
        self.layer_3 = tf.add(tf.matmul(layer_2, self.weights['w3']), self.biases['b3'])
        self.layer_drop = tf.nn.dropout(self.layer_3, self.keep_prob)
        output_layer = tf.matmul(self.layer_3, self.weights['out']) + self.biases['out']

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=output_layer))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train_neural_net(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for i in range(self.n_iterations):
            batch_x, batch_y = mnist.train.next_batch(self.batch_size)
            sess.run(self.train_step, feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: self.dropout})

            if i % 100 == 0:
                minibatch_loss, minibatch_accuracy = sess.run([self.cross_entropy, self.accuracy],
                                                              feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 1.0})

                print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))

        test_accuracy = sess.run(self.accuracy, feed_dict={self.X: mnist.test.images, self.Y: mnist.test.labels, self.keep_prob: 1.0})
        print("Accuracy on tests set:", test_accuracy)
        self.net = sess

    def recognize_image(self, img_path):
        img = imageio.imread(img_path, as_gray=True).ravel()
        img = self.convert_to_negative(img)
        output_layer = tf.matmul(self.layer_3, self.weights['out']) + self.biases['out']
        prediction = self.net.run(tf.argmax(output_layer, 1), feed_dict={self.X: [img]})
        print("Prediction for tests image:", np.squeeze(prediction))
        return np.squeeze(prediction)

    @staticmethod
    def convert_to_negative(img):
        img = np.abs(255 - img)
        return img