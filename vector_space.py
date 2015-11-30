import random
import numpy as np
import theano
import logging
import sys

from theano import tensor as T
from collections import defaultdict

# dimensions
# context window
# min freq (for words to be considered)
# calculating -ve samples
# gradient for input and output vecs

class VectorSpace:

	def __init__(self, data ,dim=100):

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)

		self.logger.info('initializing vector space...')
		# main vars
		self.vocab, self.contexts, occ = data
		self.neg_contexts = defaultdict(list)
		self.dim = dim
		self.probs = np.array(map(lambda x: x / len(occ), occ)).astype(np.float)

		self.word2vec()



	def word2vec(self, training_steps=1000, alpha=0.01):

		in_vec  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,(len(self.vocab), self.dim)) \
				.astype(theano.config.floatX))
		out_vec  = theano.shared(0.2 * np.random.uniform(-1.0, 1.0,(len(self.vocab), self.dim)) \
				.astype(theano.config.floatX))

		self.logger.info('constructing cost function...')
		u, v, n = T.vectors('u', 'v', 'n')
		t,j  = T.iscalars('t','j')
		ctx_term = T.log(1 / (1 + T.exp(-T.dot(u,v))))
		neg_term = T.log(1 / (1 + T.exp(-T.dot(n,u)))).sum()
		cost = - (ctx_term + neg_term.sum())
		gu, gv = T.grad(cost, [u, v])


		self.logger.info('compiling cost function...')




		cost_function = theano.function(inputs=[t,j,u,v,n], \
										outputs=cost, \
										updates=(	(out_vec, T.set_subtensor(out_vec[j], u - alpha * gu)), \
													(in_vec, T.set_subtensor(in_vec[t], v - alpha * gv))))





		self.logger.info('cost function compiled successfully...')
		sys.exit(0)
		self.logger.info('started training...')

		for i in range(training_steps):

			self.logger.info('training iteration: ' + str(i))

			easy_neg_samples()
			for word in self.contexts:

				cx_words = self.contexts[word]
				neg_words = self.neg_samples(word)

				for cw in cx_words:
					curr = cost_function(cw, word,in_vec[self.sum_neg_samples[cw]])


	# easier version of choosing k, just by randomly
	# choosing words not in the context windows
	def easy_neg_samples(k=5):

		self.sum_neg_samples = []
		for word in self.contexts:

			if not len(self.neg_contexts):

				neg_words = set(range(len(self.vocab))) - set(self.contexts[word])

				# fill in missing neg samples
				i=0
				while len(neg_words < 2*k):
					if i not in neg_words:
						neg_words.update(i)
					i += 1
				self.neg_contexts[word] = neg_words

			random.shuffle(self.neg_contexts[word])



	# following the definition in the word2vec paper
	def neg_samples(k=5):
		self.sum_neg_samples = np.zeros((len(self.vocab),self.dim))

		z = 0.1 # normalizing factor

		for word in range(len(self.contexts)):

			# to avoid computing U(k)^3/4 every iteration, a bigger sample is saved
			if not len(self.neg_contexts):

				samples = self.probs
				samples **= 3/4
				samples *= self.probs(word) / z

				samples_dict = dict(zip(range(len(neg_samples),neg_samples.tolist())))
				sorted_samples = sorted(samples_dict.items(), key=operator.itemgetter(1))
				self.neg_contexts[word] = [i[0] for i in sorted_x[:2*k]]

			# then k is selected randomly each iteration
			random.shuffle(self.neg_contexts[word])




