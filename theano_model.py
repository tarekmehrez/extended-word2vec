import theano


import numpy as np
from theano import tensor as T

class TheanoModel:

	def __init__(self,logger,vocab, args, data):
		self._logger = logger

		self._logger.info('initializing theano model')
		self._opt_speed()
		dim, self._epochs, self._batch_size, self._alpha, self._reg = args

		self._in_vecs = theano.shared(np.random.uniform(-1.0, 1.0, (len(vocab),dim)).astype(theano.config.floatX))
		self._out_vecs = theano.shared(np.random.uniform(-1.0, 1.0, (len(vocab),dim)).astype(theano.config.floatX))
		self._tokens = theano.shared(data[0])
		self._windows = theano.shared(data[1])
		self._neg_samples = theano.shared(data[2])
	def _cost(self,cen_idx,ctx_idx,neg_idx):

		t = self._in_vecs[cen_idx]
		j = self._out_vecs[ctx_idx]
		n = self._out_vecs[neg_idx]

		ctx_term = 	T.log(T.nnet.sigmoid(T.dot(t, j.T)))
		neg_term = T.sum(T.log(T.nnet.sigmoid(-T.dot(t, n.T))))

		return T.sum(ctx_term + neg_term)

	def compile(self):
		self._logger.info('compiling theano function')

		result, _ = theano.scan(fn=self._cost, sequences=[self._tokens,self._windows,self._neg_samples])
		self._f = theano.function([], T.sum(result))
		self._f.trust_input = True


	def _opt_speed(self):

		self._logger.info('activating theano optimizations')
		theano.config.mode = 'FAST_RUN'
		theano.config.linker = 'cvm_nogc'
		theano.config.allow_gc = False

	def train(self):

		self._logger.info('started training model')

		for epoch in range(self._epochs):
			self._logger.info('starting epoch: ' + str(epoch))
			cost = self._f.fn()
			self._logger.info('cost: ' + str(cost))

		self._logger.info('done training model')

	def _save_model(self, step):
		file = 'theano.model.' + str(step)
		self._logger.info("writing model to " + str(file) )

		with open(file, 'wb') as f:
			cPickle.dump(self._out_vecs, f)
