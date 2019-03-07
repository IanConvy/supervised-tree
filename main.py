import tensorflow as tf
import numpy as np
import optimization as opt
import data
import graph

class Model():
    def __init__(self, data_path, digits, val_split, hparams, op_shape):
        (train_data, val_data, test_data) = data.get_data(data_path, digits, val_split)
        (self.train_images, self.train_labels) = train_data
        (self.val_images, self.val_labels) = val_data
        (self.test_images, self.test_labels) = test_data
        num_pixels = self.train_images.shape[1]
        self.graph = graph.Graph(num_pixels, op_shape)
        self.optimizer = opt.Spsa_Optimizer(self.graph, hparams)

    def evaluate_images(self, sess, image_batch):
        probabilities = self.graph.run_graph(sess, image_batch)
        return probabilities

    def run_epoch(self, sess, epoch, batch_size):
        batch_iter = data.batch_generator(self.train_images, self.train_labels, batch_size)
        for (images, labels) in batch_iter:
            self.optimizer.update_params(sess, epoch, images, labels)
        val_results = self.evaluate_images(sess, self.val_images)
        val_accuracy = get_accuracy(val_results, self.val_labels)
        return val_accuracy

    def train_network(self, epochs, batch_size):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            sess.run(self.graph.init)
            for epoch in range(epochs):
                val_accuracy = self.run_epoch(sess, epoch, batch_size)
                print('{} : {:.3f}'.format(epoch, val_accuracy))
            test_accuracy = self.run_test_data(sess)
            print('Test Accuracy : {:.3f}'.format(test_accuracy))
        writer.close()

    def run_test_data(self, sess):
        test_results = self.evaluate_images(sess, self.test_images)
        test_accuracy = get_accuracy(test_results, self.test_labels)
        return test_accuracy

def get_accuracy(guesses, labels):
    guess_index = np.argmax(guesses, axis = 1)
    label_index = np.argmax(labels, axis = 1)
    compare = guess_index - label_index
    num_correct = np.sum(compare == 0)
    total = guesses.shape[0]
    accuracy = num_correct / total
    return accuracy

data_path = 'data/8'
digits = (6, 7)
val_split = 0.85
op_shape = (2, 2, 1, 2, 2)
epochs = 30
batch_size = 222
opt_params ={
  'a': 28,
  'b': 33,
  'A': 74.1,
  's': 4.13,
  't': 0.658,
  'gamma': 0.882,
  'lmbda': 0.234,
  'eta': 5.59}

model = Model(data_path, digits, val_split, opt_params, op_shape)
model.train_network(epochs, batch_size)
