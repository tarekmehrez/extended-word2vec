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


# TODO:



class VectorSpace:

	def __init__(self, corpus ,dim=100):


		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)

		self.logger.info('initializing vector space...')

		self.dim = dim

		self.neg_contexts = defaultdict(list)
		# unigrams
		self.unigrams = np.array(map(lambda x: x / len(self.corpus.freq), self.corpus.freq)).astype(np.float)
		# self.easy_init_samples()

		self.init_vectors()
		for i in range(1000):

			print i, self.calculate_cost()


	def init_vectors(self):
		self.vec_in = np.random.uniform(0,1.0,(len(self.corpus.vocab),self.dim))
		self.vec_out = np.random.uniform(0,1.0,(len(self.corpus.vocab),self.dim))


	def train(self):

		alpha = 0.1
		training_iter = 1000

		inside_words = self.corpus.vocab
		co_occ = self.corpus.cocc_matrix
		# self.shuffle_samples()

		cost = 0
		for t, in_word in enumerate(inside_words):


			for j, ctx_word in enumerate(co_occ[t]):

				i = get_samples(j)

				delta_j = 1 - sigmoid(np.dot(vec_out[j],vec_in[t])) * vec_in[t]
				delta_j += np.sum(1 - sigmoid(np.dot(vec_out[j],vec_in[i])) * vec_in[i])

				delta_t = 1 - sigmoid(np.dot(vec_out[j],vec_in[t])) * vec_in[j]

				vec_out[j] -= alpha * delta_j
				vec_in[t] -= alpha * delta_t

	def calculate_cost(self):

		inside_words = self.corpus.vocab
		co_occ = self.corpus.cocc_matrix
		# self.shuffle_samples()

		cost = 0
		for t, in_w in enumerate(inside_words):


			for j, c_w in enumerate(co_occ[t]):

				co_entry = co_occ[t][j]

				n_mat = self.vec_in[self.get_samples(c_w)]
				c_mat = np.resize(self.vec_out[j], n_mat.shape)

				c_vec = self.vec_out[j]
				t_vec = self.vec_in[t]

				cost += (np.log(self.sigmoid(np.dot(c_vec.T,t_vec))))

				cost += (np.log(self.sigmoid(np.dot(c_mat.T,n_mat)))).sum()

		cost /= len(inside_words)

		return cost

	def derivative()

	def sigmoid(self,x):
		return 1/ (1 - np.exp(-x))

	# easier implementation of getting neg samples
	def easy_init_samples(self,k=5):

		for word in self.corpus.cocc_matrix:

			neg_words = np.where(word==0)[0]
			print len(neg_words)
			if len(neg_words) is 0:
				neg_words = []

			# fill in missing neg samples
			i=0
			while len(neg_words < 2*k):
				if i not in neg_words:
					neg_words.append(i)
				i += 1
			self.neg_contexts[word] = neg_words



	# following the definition in the word2vec paper
	def init_samples(k=5):

		z = 0.1 # normalizing factor

		for word, entry in enumerate(self.corpus.cocc_matrix):

			# to avoid computing U(k)^3/4 every iteration, a bigger sample is saved

			samples = self.unigrams
			samples **= 3/4
			samples *= self.unigrams(word) / z

			samples_dict = dict(zip(range(len(neg_samples),neg_samples.tolist())))
			sorted_samples = sorted(samples_dict.items(), key=operator.itemgetter(1))
			self.neg_contexts[word] = [i[0] for i in sorted_x[:2*k]]

			# then k is selected randomly each iteration

	def shuffle_samples(self):
		for word in self.neg_contexts:
			random.shuffle(self.neg_contexts[word])

	def get_samples(self,word, k=5):
		return np.random.randint(size=k, low=0, high=len(self.corpus.vocab))
