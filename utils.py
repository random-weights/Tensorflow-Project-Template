import json
from bunch import Bunch
import os


def write_to_json(exp_name, epochs, iter_per_epoch, batch_size, learning_rate):
	"""
	Makes sense to store each config file inside the experiments/exp_name dir.
	That way all the data regarding an experiment is in one directory.
	this config file will be generated for each instance of trainer obj.
	"""
	edict = {
		"exp_name": exp_name,
		"epochs": epochs,
		"iter_per_epoch": iter_per_epoch,
		"batch_size": batch_size,
		"learning_rate": learning_rate}


def create_dirs(dirs):
	"""
	dirs - a list of directories to create if these directories are not found
	:param dirs:
	:return exit_code: 0:success -1:failed
	"""
	try:
		for dir_ in dirs:
			if not os.path.exists(dir_):
				os.makedirs(dir_)
		return 0
	except Exception as err:
		print("Creating directories error: {0}".format(err))
		exit(-1)