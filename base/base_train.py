"""
some attributes of trainer class are hyper params for model
1. epochs
2. iter per epochs
3. learning rate
4. batch_size
"""
import tensorflow as tf


class BaseTrain:
	def __init__(self, sess, model, train_data,epochs,iter_per_epoch,batch_size):
		self.model = model
		self.sess = sess
		self.data = train_data
		self.epochs = epochs
		self.iter_per_epoch = iter_per_epoch
		self.batch_size = batch_size

		self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(self.init)

	def save_train_details(self,name):
		pass

	def train(self):
		"""
		implement logic of training process,
		-what to do after/certain no. of epoch(s)
		-any console output or progress bar
		"""
		raise NotImplementedError

	def train_epoch(self):
		"""
		implement the logic of epoch:
		-loop over the number of iterations in the config and call the train step
		-add any summaries you want using the summary
		"""
		raise NotImplementedError

	def train_step(self):
		"""
		implement the logic of the train step
		- run the tensorflow session
		- return any metrics you need to summarize
		"""
		raise NotImplementedError
