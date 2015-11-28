import os
import json
import logging

import numpy as np
from collections import defaultdict


class Corpus:

	def __init__(self, input_dir, entities, cw):

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)

		self.cw = cw
		self.entities = entities
		self.read_input(input_dir)
		self.make_contexts()
		self.write()



	# read input, build vocab
	def read_input(self, input_dir):
		self.logger.info("Now reading from input dir: " + input_dir)

		vocab = set()
		self.sources_content = []
		for i, file in enumerate(os.listdir(input_dir)):
			f = open('art-data/' + file, 'r')

			# assuming it's a one line file with no punctuations
			# and tokens are already space separated

			current = f.read().strip().split(' ')
			current = np.array(current)

			for e in self.entities:
				indices = np.where(current==e)[0]
				current[indices] = e + '_' + str(i)

			vocab.update(current)
			self.sources_content.append(current)

		self.logger.info("Constructing vocab indices...")
		self.vocab = dict(zip(dict.fromkeys(vocab),range(len(vocab))))

	def make_contexts(self):

		self.logger.info("Constructing context windows...")


		for i, source in enumerate(self.sources_content):

			# replace words by indices
			for word in self.vocab:
				indices = np.where(source == word)[0]
				self.sources_content[i][indices] = self.vocab[word]

			# change idx from strings to ints
			self.sources_content[i] = self.sources_content[i].astype(np.int)

			# build context windows
			contexts = defaultdict(list)

			# to handle edges
			upper = lambda x: 0 if x < 0 else x

			for j,word in enumerate (self.sources_content[i]):
				curr = self.sources_content[i][upper(j-self.cw):j+self.cw+1]

				# delete current word from context list
				mask = np.ones(len(curr), np.bool)
				indices = np.where(curr==word)[0]
				mask[indices]=False
				curr = curr[mask]

				contexts[word] += curr.tolist()
		self.contexts = contexts

	def write(self):
		self.logger.info("Writing data to json files...")

		with open('vocab.json', 'w') as outfile:
			json.dump(self.vocab, outfile)

		with open('contexts.json', 'w') as outfile:
			json.dump(self.contexts, outfile)

		self.logger.info("done writing!")


	def read(self):
		with open('vocab.json') as json_data:
			self.vocab = json.load(json_data)


		with open('contexts.json', 'w') as outfile:
			self.cotexts = json.load(contexts, outfile)

		return (self.vocab, self.contexts)