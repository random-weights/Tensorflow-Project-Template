import json
from bunch import Bunch
import os

"""
Makes sense to store each config file inside the experiments/exp_name dir.
That way all the data regarding an experiment is in one directory.
this config file will be generated for each instance of trainer obj.
"""


class Experiment:
	def __init__(self,exp_name):
		self.exp_name = exp_name
		self.epochs = None
		self.iter_per_epoch = None
		self.batch_size = None
		self.learning_rate = None

	def write_to_json(self):
		edict = {
			"exp_name": self.exp_name,
			"epochs": self.epochs,
			"iter_per_epoch": self.iter_per_epoch,
			"batch_size": self.batch_size,
			"learning_rate": self.learning_rate}
