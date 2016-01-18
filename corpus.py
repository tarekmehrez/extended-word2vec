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
		self._freq['<PADDING_TOKEN>']=0
		self._logger.info('corpus size: ' + str(self.get_corpus_size()))
		self._logger.info('vocab size: ' + str(len(self._freq)))

	def _create_ctx_windows(self):

		self._logger.info('creating ctx windows')

		self._windows = []
		self._tokens = []

		vocab = self._freq.keys()

		for csv_row in self._source_files:

			curr = self._read_file(csv_row)
			idx = map(lambda word: vocab.index(word), curr)

			# TODO add indicator <end of file>
			self._context_win(idx)

		self._windows = np.array(self._windows, dtype=np.int32)
		self._tokens = np.array(self._tokens, dtype=np.int32)

	def _read_file(self, csv_row):

		# TODO handle punctuations, stop, frequent and rare words
		file = csv_row[0]
		source = csv_row[1]

		f = open(self._dir + '/' + file, 'r')
		curr = f.read().strip()

		for e in self._entities:
			new = e + '_' + source
			curr = curr.replace(e, new)

		curr = curr.split(' ')

		return curr



	def _context_win(self,input):
		# input = input[:1000]
		lpadded = self._cw // 2 * [-1] + input + self._cw // 2 * [-1]
		self._windows += [lpadded[i:(i + self._cw)] for i in range(len(input))]
		self._tokens += input



	def _create_neg_samples(self, samples=5):

		self._logger.info('creating neg samples')


		idx = range(len(self._freq))
		freq = np.array(self._freq.values(), dtype=np.float)
		freq = np.power(freq / np.sum(freq), 0.75) # unigrams ^ 3/4
		dist = freq * (1 / np.sum(freq)) #normalize probabs

		self._neg_samples = np.zeros((len(self._windows), samples + self._cw), dtype=np.int32)

		for example in  range(len(self._neg_samples)):
			self._neg_samples[example] = np.random.choice(idx, (samples + self._cw), p=dist)


	def _save(self):
		self._logger.info("saving corpus object to corpus.pkl")
		self._logger = None

		with open('corpus.pkl', 'wb') as f:
			cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)


	def get_data(self):
		return self._tokens, self._windows, self._neg_samples

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