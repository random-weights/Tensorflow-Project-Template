from base import BaseTrain
from tqdm import tqdm
import numpy as np


class ExampleTrainer(BaseTrain):
	def __init__(self,name,sess,model,data,epochs,iter_per_epoch,batch_size):
		super(ExampleTrainer, self).__init__(sess,model,data,epochs,iter_per_epoch,batch_size)
		self.save_train_details(name)

	def train(self):
		loop = tqdm(range(self.epochs))
		for _ in loop:
			self.train_epoch()

	def train_epoch(self):
		loop = tqdm(range(self.iter_per_epoch))
		losses = []
		accs = []
		for _ in loop:
			loss, acc = self.train_step()
			losses.append(loss)
			accs.append(acc)
		loss = np.mean(losses)
		acc = np.mean(accs)

		self.model.save(self.sess)

	def train_step(self):
		batch_x, batch_y = next(self.data.next_batch(self.batch_size))
		feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
		_, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
									 feed_dict=feed_dict)
		return loss, acc
