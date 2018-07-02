"""
some attributes of trainer class are hyper params for model
1. epochs
2. iter per epochs
3. learning rate
"""

import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, data):
        self.model = model
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self,epochs):
        """
        :param epochs: no. of passes through whole training set
        :return: None
        """
        raise NotImplementedError

    def train_epoch(self,iter_per_epoch):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self,batch_size):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
