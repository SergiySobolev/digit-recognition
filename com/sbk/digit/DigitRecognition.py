import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image


class DigitRecognition:

    image_width = 28
    image_height = 28
    n_input = image_width * image_height
    n_hidden1 = 512
    n_hidden2 = 256
    n_hidden3 = 128
    n_output = 10

    learning_rate = 1e-4
    n_iterations = 1200
    batch_size = 2048
    dropout = 0.5

    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])
    keep_prob = tf.placeholder(tf.float32)

    weights = {
        'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
        'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
        'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
        'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
    }

    layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    layer_drop = tf.nn.dropout(layer_3, keep_prob)
    output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
        print("Accuracy on test set:", test_accuracy)
        self.net = sess

    def recognize_image(self, img_path):
        img = np.invert(Image.open(img_path).convert('L')).ravel()
        output_layer = tf.matmul(self.layer_3, self.weights['out']) + self.biases['out']
        prediction = self.net.run(tf.argmax(output_layer, 1), feed_dict={self.X: [img]})
        print("Prediction for test image:", np.squeeze(prediction))

        return np.squeeze(prediction)