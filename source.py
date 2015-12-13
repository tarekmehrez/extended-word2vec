import os
import logging

import numpy as np

class Source:

	def __init__(self, file, entities,name):

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)
		self.name = name

		self.logger.info("creating source: "+str(self.name))


		f = open(file, 'r')
		curr = f.read().strip()

		for e in entities:
			curr = curr.replace(e, e + '_' + str(self.name))

		curr = np.array(curr.split(' '))
		self.content = curr
		self.vocab = set(curr)

	def get_vocab(self):
		return self.vocab

	def get_content(self):
		return self.content


