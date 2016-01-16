import cPickle, sys

import gensim
import theano
import numpy as np

from theano import tensor as T
from theano_model import TheanoModel

class VectorSpace:

	def __init__(self, logger, args):

		self._logger = logger
		self._load_corpus()


		self._logger.info('initializing vector space')


		self._args = args

		self._freq = self._corpus.get_freq()
		self._vocab = self._corpus.get_vocab()
		self._windows = self._corpus.get_windows()
		self._neg_samples = self._corpus.get_neg_samples()


		# self._train_gensim()
		# self._train_raw()
		self._train_theano()


	def _train_gensim(self):
		model = GensimModel(logger, self._corpus)
		model.train()

	def _train_theano(self):

		model = TheanoModel(self._logger,self._vocab, self._args)
		model.compile()

		tokens, ctx, negs = self._corpus.get_data()
		model.train(tokens, ctx, negs)



	def _load_corpus(self):
		self._logger.info("loading corpus object from corpus.pkl")

		with open('corpus.pkl', 'r') as f:
			self._corpus = cPickle.load(f)



