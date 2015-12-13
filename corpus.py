import logging
import os

import numpy as np
import cPickle as pickle

from source import Source
from collections import defaultdict

class Corpus:

	def __init__(self,input_dir, entities):

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)


		self.filename = 'corpus.pkl'
		if os.path.exists(self.filename):
			self.load()
		else:
			self.read_input(input_dir, entities)

	# reads in vocab, replaces entites by entities_source they belong to
	def read_input(self, input_dir,entities):

		self.sources = []
		vocab = set()

		self.logger.info('started reading input from: ' + str(input_dir))

		for i, file in enumerate(os.listdir(input_dir)):

			# TODO replace i with the real source name

			s = Source(input_dir + '/' + file,entities,i)
			self.sources.append(s)
			vocab.update(s.get_vocab())

		self.vocab = list(vocab)
		self.logger.info('done reading input')
		self.logger.info('vocab size = ' + str(len(self.vocab)))
		self.create_ctx()

	# gets context words for each word in vocab, creates word freq for all sources
	def create_ctx(self, cw=3):

		freq = np.zeros(len(self.vocab))
		ctx_words = defaultdict(list)

		self.logger.info("Getting context words and counting word frequencies...")

		for source in self.sources:

			curr = source.get_content().tolist()

			# replace words by indices
			curr = map(lambda word: self.vocab.index(word), curr)

			# build context windows
			upper = lambda x: 0 if x < 0 else x

			for j,word in enumerate(curr):

				freq[word] += 1
				contexts = curr[upper(j-cw):j+cw+1]
				ctx_words[word] = contexts.remove(word)

		freq = np.array(freq).astype(np.int)
		self.freq = freq
		self.ctx_words = ctx_words
		self.logger.info("done with contexts...")
		self.logger.info("corpus size: " + str(self.freq.sum()) + " tokens")
		self.save()


	# TODO, pickle and load self.sources

	def load(self):
		self.logger.info("Loading corpus...")

		f = open(self.filename,'rb')
		data = pickle.load(f)
		self.vocab, self.ctx_words, self.freq = data


	def save(self):
		self.logger.info("Writing corpus")
		f = open(self.filename,'wb')
		data = [self.vocab, self.ctx_words, self.freq]
		pickle.dump(data,f)
		f.close()



	def idx2word(self, list):
		new_vocab = dict (zip(self.vocab.values(),self.vocab.keys()))
		return map(lambda idx: new_vocab[idx], list)
