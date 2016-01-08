import random
import numpy as np
import theano
import logging
import sys

from theano import tensor as T
from collections import defaultdict




class VectorSpace:

	# TODO enter dim as user input

	def __init__(self, corpus ,dim=100):


		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)

		self.logger.info('initializing vector space...')


		self.corpus = corpus
		self.vec_in = np.random.uniform(0,1.0,(len(self.corpus.vocab),dim))
		self.vec_out = np.random.uniform(0,1.0,(len(self.corpus.vocab),dim))
		self.train()

	# TODO enter alpha and training iterations as user input

	def train(self, alpha=0.1, iter=1000):

		self.logger.info('starting training...')

		time_steps = self.corpus.vocab
		ctx_words = self.corpus.ctx_words
		sigmoid = lambda x: 1/(1+np.exp(-x))


		self.shuffle_samples()


		for t, in_word in enumerate(time_steps):
			self.logger.info('time step ' + str(t))

			for j, co_occ in enumerate(ctx_words[t]):

				n = self.get_samples(j)
				delta_j = 1 - sigmoid(np.dot(self.vec_out[j],self.vec_in[t])) * self.vec_in[t]

				delta_t = 	1 - sigmoid(np.dot(self.vec_out[j],self.vec_in[t])) * self.vec_in[j] + \
							np.sum(1 - sigmoid(np.dot(-self.vec_in[t],self.vec_out[n].T)).T * -self.vec_in[n])

				self.vec_out[j] -= alpha * ( delta_j * co_occ)
				self.vec_in[t] -= alpha * ( delta_t * co_occ)



	def shuffle_samples(self):
		for word in self.corpus.neg_samples:
			random.shuffle(self.corpus.neg_samples[word])

	def get_samples(self,word, k=5):
		return self.corpus.neg_samples[word][:k]
