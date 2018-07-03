import tensorflow as tf


class BaseModel:
	def __init__(self, config):
		self.config = config
		self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
		self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

	# save function that saves the checkpoint in the path defined in the config file
	def save(self, sess):
		print("Saving model...")
		self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
		print("Model saved")

	# load latest checkpoint from the experiment path defined in the config file
	def load(self, sess):
		latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
		if latest_checkpoint:
			print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
			self.saver.restore(sess, latest_checkpoint)
			print("Model loaded")

	def build_model(self):
		raise NotImplementedError


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
