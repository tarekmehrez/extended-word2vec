import logging


import csv

import cPickle as pickle

from collections import defaultdict

from nltk import FreqDist

# reads in corpus, must have files, sources.csv and entities.txt

class Corpus:

	def __init__(self,args):

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self._logger = logging.getLogger(__name__)
		self._dir = args[0]

		self._read_dir()
		self._create_vocab()

	def _read_dir(self):

		self._logger.info('reading source files & named entities...')


		with open(self._dir + '/sources.csv','r') as f:
			reader = csv.reader(f)
			files = [files for files in reader]
		self._source_files = files

		with open(self._dir + '/entities.txt','r') as f:
			reader=f.read()
		self._entities = reader.split('\n')[:-1]



	def _create_vocab(self):

		self._logger.info('creating vocab')

		self._freq = defaultdict(int)


		for csv_row in self._source_files:
			curr = self._read_file(csv_row)
			for i in curr:
				self._freq[i] += 1


		self._logger.info('vocab size = ' + str(len(self._freq)))


	def _read_file(self, csv_row):

		# TODO clean punctuations, remove stop, frequent and rare words
		file = csv_row[0]
		source = csv_row[1]

		f = open(self._dir + '/' + file, 'r')
		curr = f.read().strip()

		for e in self._entities:
			new = e + '_' + source
			curr = curr.replace(e, new)

		return curr.split(' ')


