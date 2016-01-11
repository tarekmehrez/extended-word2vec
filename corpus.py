import os, csv, sys, cPickle

import numpy as np

from collections import defaultdict
from collections import OrderedDict
from scipy import stats

# reads in corpus, must have files, sources.csv and entities.txt

class Corpus:

	def __init__(self,logger, args):

		self._logger = logger

		if os.path.exists('corpus.pkl'):
			self._logger.info('corpus.pkl already exists')
			sys.exit(1)

		self._dir, self._cw = args
		self._read_dir()
		self._create_vocab()
		self._create_ctx_windows()
		self._create_neg_samples()
		self._save()

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


		self._freq = OrderedDict(sorted(self._freq.items()))

		self._logger.info('corpus size: ' + str(self.get_corpus_size()))
		self._logger.info('vocab size: ' + str(len(self._freq)))


	def _create_ctx_windows(self):

		self._logger.info('creating ctx windows')

		self._windows = []
		vocab = self._freq.keys()

		for csv_row in self._source_files:

			curr = self._read_file(csv_row)
			idx = map(lambda word: vocab.index(word), curr)

			# TODO add indicator <end of file>
			self._windows.append(self._context_win(idx))


	def _read_file(self, csv_row):

		# TODO handle punctuations, stop, frequent and rare words
		file = csv_row[0]
		source = csv_row[1]

		f = open(self._dir + '/' + file, 'r')
		curr = f.read().strip()

		for e in self._entities:
			new = e + '_' + source
			curr = curr.replace(e, new)

		return curr.split(' ')


	def _context_win(self,input):

		padding = lambda x: 0 if x < 0 else x
		windows = []
		for idx in input:
			windows += input[ padding ( idx - self._cw ) : idx + self._cw + 1 ]

		return windows


	def _create_neg_samples(self, samples=5):

		self._logger.info('creating neg samples')


		freq = np.array(self._freq.values(), dtype=float)
		idx = range(len(self._freq.keys()))
		freq = np.power(freq / np.sum(freq), 3/4)
		neg_dist = stats.rv_discrete(name='_neg_dist', values=(idx, freq))

		self._neg_samples = []
		for i in self._windows:
			self._neg_samples += neg_dist.rvs(size=samples).tolist()


	def _save(self):
		self._logger.info("saving corpus object to corpus.pkl")
		self._logger = None

		with open('corpus.pkl', 'wb') as f:
			cPickle.dump(self, f)



	def get_neg_samples(self):
		return self._neg_samples

	def get_windows(self):
		return self._windows

	def get_corpus_size(self):
		return sum(self._freq.values())

	def get_vocab(self):
		return self._freq.keys()

	def get_freq(self):
		return self._freq

	def get_source_files(self):
		return self._source_files