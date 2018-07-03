from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class TemplateTrainer(BaseTrain):
	def __init__(self,name,sess,model,data,epochs,iter_per_epoch,batch_size):
		super(TemplateTrainer,self).__init__(sess,model,data,epochs,iter_per_epoch,batch_size)
		self.save_train_details(name)

	def train(self):
		pass

	def train_epoch(self):
		pass

	def train_step(self):
		pass

	def save_train_details(self,name):
		pass
