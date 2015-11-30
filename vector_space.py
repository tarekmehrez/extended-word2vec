import theano
import random
import numpy as np

from theano import tensor as T
from collections import defaultdict

# dimensions
# context window
# min freq (for words to be considered)
# calculating -ve samples
# gradient for input and output vecs

class VectorSpace:

	def __init__(self, vocab, contexts, occ ,dim=100):
		self.vocab = vocab
		self.contexts = contexts
		self.neg_contexts = defaultdict(list)
		self.dim  = dim
		self.word2vec()
		self.probs = np.array(map(lambda x: x / len(occ), occ)).astype(np.float)

	def word2vec(self, training_steps=100):

		self.in_vec = 0.2 * numpy.random.uniform(-1.0, 1.0,(len(self.vocab), self.dim))
		self.out_vect = 0.2 * numpy.random.uniform(-1.0, 1.0,(len(self.vocab), self.dim))


	# easier version of choosing k, just by randomly
	# choosing words not in the context windows
	def easy_neg_samples(self, k=5):

		self.sum_neg_samples = np.zeros((len(self.vocab),self.dim))
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
			self.sum_neg_samples[word] = self.in_vec(self.neg_contexts[word][:k]).sum()


	# following the definition in the word2vec paper
	def neg_samples(self, k=5):
		self.sum_neg_samples = np.zeros((len(self.vocab),self.dim))

		z = 0.1 # normalizing factor

		for word in range(len(self.contexts)):

			# to avoid computing U(k)^3/4 every iteration, a bigger sample is saved
			if not len(self.neg_contexts):

				neg_samples = self.probs
				neg_samples **= 3/4
				neg_samples *= self.probs(word) / z

				samples_dict = dict(zip(range(len(neg_samples),neg_samples.tolist())))
				sorted_samples = sorted(samples_dict.items(), key=operator.itemgetter(1))
				self.neg_contexts[word] = [i[0] for i in sorted_x[:2*k]]

			# then k is selected randomly each iteration
			random.shuffle(self.neg_contexts[word])
			self.sum_neg_samples[word] = self.in_vec(self.neg_contexts[word][:k]).sum()



