import cPickle, sys

import gensim
import theano
import numpy as np

from theano import tensor as T


class VectorSpace:

	def __init__(self, logger, args):

		self._logger = logger
		self._load_corpus()


		self._logger.info('initializing vector space')


		self._dim, self._epochs, self._batch_size, self._alpha, self._reg = args

		self._freq = self._corpus.get_freq()
		self._vocab = self._corpus.get_vocab()
		self._windows = self._corpus.get_windows()
		self._neg_samples = self._corpus.get_neg_samples()


		# self._train_gensim()
		# self._train_raw()
		self._train_theano()


	# basic word2vec in gensim (benchmark)

	def _train_gensim(self):
		self._logger.info('starting training vectors with gensim')

		tokens = []
		for csv_row in self._corpus.get_source_files():
			tokens += self._corpus._read_file(csv_row)

		sentences = []
		offset = 10
		curr = []

		for i, t in enumerate(tokens):

			curr.append(t)
			offset -= 1

			if offset == 0:
				sentences.append(curr)
				offset = 10
				curr = []

		model = gensim.models.Word2Vec()
		model.build_vocab(sentences)

		model.train(sentences)

		model.save('gensim.model')
		self._logger.info('done training vectors with gensim')


	# theano model of regularized word2vec

	def _train_theano(self):


		self._logger.info('initializing training with theano')

		in_vecs = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (len(self._vocab),self._dim)).astype(theano.config.floatX))
		out_vecs = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (len(self._vocab),self._dim)).astype(theano.config.floatX))

		central = T.ivector('context')
		context = T.ivector('context')
		negative= T.ivector('negative')

		t = in_vecs[central]
		j = out_vecs[context]
		i = out_vecs[negative]


		ctx_term = 	T.log(T.nnet.sigmoid(T.dot(t, j.T)))
		neg_term = T.sum(T.log(T.nnet.sigmoid(T.dot(t, i.T))))

		expr = T.sum(ctx_term + neg_term)
		grad_central, grad_context = T.grad(expr, [t, j])

		self._logger.info('compiling theano function')

		train = theano.function(inputs=[central,context,negative],outputs=[grad_central,grad_context])

		self._logger.info('done compiling theano function')

		self._logger.info('computing gradients')

		for idx, window in enumerate(self._windows):
			key = window.keys()[0]
			result = train([key], window[key], self._neg_samples[idx])
			# print result
		self._logger.info('done gradients')

	# raw implementation of regularized word2vec

	def _train_raw(self):

		self._in_vecs = np.random.rand(len(self._vocab),self._dim)
		self._out_vecs =  np.random.rand(len(self._vocab),self._dim)
		self._logger.info('starting training vectors')

		upper = lambda x: 0 if x < 0 else x

		for epoch in range(self._epochs):

			# cost = self._cost()
			# self._logger.debug('epoch: ' + str(epoch) + ', cost: ' + str(cost))

			self._logger.debug('epoch: ' + str(epoch))

			self._in_updates = np.zeros((len(self._vocab),self._dim))
			self._out_updates = np.zeros((len(self._vocab),self._dim))


			for csv_row in self._corpus.get_source_files():

				curr_file = self._corpus.words_to_idx(csv_row)
				self._logger.debug('reading file: ' + str(csv_row))
				for idx, central in enumerate(curr_file):

					window = curr_file[ upper ( idx - self._cw ) : idx + self._cw + 1 ]

					# self._logger.debug('getting grads for central: ' + str(central) +  ' and window ' + str(window))
					updates = self._grad(self._in_vecs[central], self._out_vecs[window], self._out_vecs[self._neg_samples()])

					self._in_updates[central] += updates[0] * self._alpha
					self._out_updates[window] += updates[1] * self._alpha

			self._logger.debug('performing updates')
			self._in_vecs -= self._in_updates
			self._out_vecs -= self._out_updates


			# decaying alpha
			self._alpha -= self._alpha / self._epochs

	def _grad(self,central,context, negative):

		grad_context = np.multiply(self._sig_grad(central, context),context.T).T

		grad_central = self._sig_grad(central, context)
		grad_central = np.sum(np.array(map(lambda i: i * context, grad_central)))
		grad_central += np.sum(np.multiply(self._sig_grad(central, negative),negative.T).T)

		# grad_central =  map(lambda ctx: ctx + np.sum(self._sig_grad(central,negative)), grad_context)
		return (grad_central, grad_context)

	def _sig_grad(self,x,y):
		return (1 - self._sigmoid(np.dot(x,y.T)))

	# raw implementation of cost function

	# def _cost(self):

	# 	upper = lambda x: 0 if x < 0 else x
	# 	cost = 0
	# 	for csv_row in self._corpus.get_source_files():
	# 		curr_file = self._corpus.words_to_idx(csv_row)
	# 		for idx, central in enumerate(curr_file):
	# 			window = curr_file[ upper ( idx - self._cw ) : idx + self._cw + 1 ]
	# 			ctx = self._cost_term(self._in_vecs(central), self._out_vecs(window))
	# 			neg = map(lambda i: i + self._cost_term(self._in_vecs(central), self._out_vecs(self._neg_samples())), ctx)
	# 			cost += ctx + neg

		# return cost / self.corpus.get_corpus_size()

	# def _cost_term(self,x,y):
	# 	return np.log(self._sigmoid(np.dot(x,y.T))


	def _sigmoid(self,x):
		return  1 / (1 + np.exp(-x))



	def _load_corpus(self):
		self._logger.info("loading corpus object from corpus.pkl")

		with open('corpus.pkl', 'r') as f:
			self._corpus = cPickle.load(f)



