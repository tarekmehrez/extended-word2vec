import os
import json
import logging

import numpy as np
from collections import defaultdict


class Corpus:

	def __init__(self):

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)


	# read input, build vocab
	def read_input(self, input_dir, entities):
		self.logger.info("Now reading from input dir: " + input_dir)

		vocab = set()

		self.logger.info("Constructing vocab...")

		self.content = []
		for i, file in enumerate(os.listdir(input_dir)):
			f = open('art-data/' + file, 'r')
			curr = f.read().strip()

			for e in entities:
				curr = curr.replace(e, e + '_' + str(i))

			curr = np.array(curr.split(' '))
 			vocab.update(curr)
			self.content.append(curr)

		self.vocab = dict(zip(dict.fromkeys(vocab),range(len(vocab))))



	def make_contexts(self, cw):
		self.logger.info("Constructing context windows...")
		contexts = defaultdict(list)

		for i, source in enumerate(self.content):

			# replace words by indices
			for word in self.vocab:
				indices = np.where(source == word)[0]
				self.content[i][indices] = self.vocab[word]

			# change idx from strings to ints
			self.content[i] = self.content[i].astype(np.int)

			# build context windows

			# to handle edges
			upper = lambda x: 0 if x < 0 else x
			for j,word in enumerate (self.content[i]):

				curr = self.content[i][upper(j-cw):j+cw+1]
				contexts[word] += curr.tolist()
				# delete current word from context list
				# mask = np.ones(len(curr), np.bool)
				# indices = np.where(curr==word)[0]
				# mask[indices]=False
				# curr = curr[mask]


		self.contexts = contexts
	def write(self):
		self.logger.info("Writing data to json files...")

		with open('data.json', 'w') as f:
			json.dump((self.vocab,self.contexts), f)

		self.logger.info("done writing!")


	def read(self):
		self.logger.info("Reading data from json files...")

		with open('data.json') as f:
			return json.load(f)

	def idx2word(self, list):
		for word in self.vocab:
			indices = np.where(list == self.vocab[word])[0]
			list[indices] = word
