import numpy as np
import tensorflow as tf
import graph

class Spsa_Optimizer():
    def __init__(self, graph, hparams):
        self.a = hparams['a']
        self.b = hparams['b']
        self.A = hparams['A']
        self.s = hparams['s']
        self.t = hparams['t']
        self.gamma = hparams['gamma']
        self.lmbda = hparams['lmbda']
        self.eta = hparams['eta']
        self.graph = graph
        self.create_loss_ops()
        self.pert_size = self.graph.num_op_params * len(self.graph.all_ops)
        self.old_update = np.zeros(self.pert_size)
        self.guess_list = []

    def create_loss_ops(self):
        self.labels = tf.placeholder(tf.float64)
        self.guesses = tf.placeholder(tf.float64)
        remove_false = tf.multiply(self.guesses, self.labels)
        prob_true = tf.reduce_max(remove_false, axis = 1)
        remove_true = self.guesses - self.labels
        highest_prob_false = tf.reduce_max(remove_true, axis = 1)
        shifted_difference = highest_prob_false - prob_true + self.lmbda
        loss_intermediate = tf.maximum(tf.cast(0, tf.float64), shifted_difference)
        self.loss = tf.reduce_mean(tf.pow(loss_intermediate, self.eta))

    def get_loss(self, sess, images, labels, pert):
        sess.run(self.graph.update_params, feed_dict = {self.graph.update : pert})
        guesses = self.graph.run_graph(sess, images)
        fd_dict = {self.guesses : guesses, self.labels:labels}
        loss = sess.run(self.loss, feed_dict = fd_dict)
        sess.run(self.graph.update_params, feed_dict = {self.graph.update : -pert})
        return loss

    def update_params(self, sess, epoch, images, labels):
        (alpha, beta) = self.anneal_hparams(epoch)
        bern_pert = 2 * np.round(np.random.random(self.pert_size)) - 1
        plus_pert = alpha * bern_pert
        minus_pert = -alpha * bern_pert
        plus_loss = self.get_loss(sess, images, labels, plus_pert)
        minus_loss = self.get_loss(sess, images, labels, minus_pert)
        gradient = (plus_loss - minus_loss) / (2 * alpha * bern_pert)
        new_update = self.gamma * self.old_update - beta * gradient
        sess.run(self.graph.update_params, feed_dict = {self.graph.update:new_update})
        self.old_update = new_update

    def anneal_hparams(self, epoch):
        a_anneal = (epoch + 1 + self.A) ** self.s
        b_anneal = (epoch + 1) ** self.t
        alpha = self.a / a_anneal
        beta = self.b / b_anneal
        return (alpha, beta)
