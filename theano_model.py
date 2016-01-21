import cPickle, os, sys
import theano

import numpy as np
from theano import tensor as T

class TheanoModel:

	def __init__(self,logger,vocab, args, ):
		self._logger = logger

		self._logger.info('initializing theano model')
		# self._opt_speed()
		dim, self._epochs, self._batch_size, self._alpha, self._reg = args[1:]

		self._in_vecs = theano.shared(np.random.uniform(-1.0, 1.0, (len(vocab),dim)).astype(theano.config.floatX))
		self._out_vecs = theano.shared(np.random.uniform(-1.0, 1.0, (len(vocab),dim)).astype(theano.config.floatX))



	def _cost(self,cen_idx,ctx_idx,neg_idx,e_idx, parallel_e_idx):

		t = self._in_vecs[cen_idx]
		j = self._out_vecs[ctx_idx]
		n = self._out_vecs[neg_idx]

		ctx_term = 	T.log(T.nnet.sigmoid(T.dot(t, j.T)))
		neg_term = T.sum(T.log(T.nnet.sigmoid(-T.dot(t, n.T))))


		reg_term = self._regularizer(e_idx, parallel_e_idx)
		cost = T.sum(ctx_term + neg_term) + (self._reg * T.sum(reg_term))

		grad_central, grad_context = T.grad(cost,[t,j])

		updates = ((self._in_vecs, T.inc_subtensor(t, (self._alpha * grad_central))), \
		(self._out_vecs, T.inc_subtensor(j, (self._alpha * grad_context))))

		return cost, updates

	def _regularizer(self, e_idx, parallel_e_idx):

		p = self._out_vecs[e_idx]
		p_prime = self._out_vecs[parallel_e_idx]

		reg_term = T.sqr(p - T.mean(p_prime))



		return reg_term


	def compile(self):
		self._logger.info('compiling theano function')

		tokens  = T.ivector('tokens')
		windows  = T.imatrix('windows')
		neg_samples  = T.imatrix('neg_samples')

		entities = T.ivector('entities')
		parallel_entities = T.imatrix('parallel_entities')


		result, updates = theano.scan(fn=self._cost, sequences=[tokens, windows, neg_samples], non_sequences=[entities, parallel_entities])

		self._f = theano.function([tokens, windows, neg_samples, entities, parallel_entities], T.mean(result))
		self._f.trust_input = True


	def _opt_speed(self):

		self._logger.info('activating theano optimizations')
		theano.config.mode = 'FAST_RUN'
		theano.config.linker = 'cvm_nogc'
		theano.config.allow_gc = False

	def train(self, sources):

		self._logger.info('started training model')

		steps = self._epochs / 5

		for epoch in range(self._epochs):

			self._logger.info('starting epoch: ' + str(epoch))

			for src_name,src in sources.iteritems():

				cost = self._f(	src.get_tokens(),
								src.get_windows(),
								src.get_neg_samples(),
								src.get_entities(),
								src.get_p_entities())

				src.shuffle_data()

			self._logger.info('cost: ' + str(cost))

			if epoch in range(steps,self._epochs,steps):
				self._save_model(epoch)

		self._save_model('final')
		self._logger.info('done training model')




	def _save_model(self, step):

		if not os.path.exists('curr-models'):
			os.makedirs('curr-models')

		file = 'curr-models/theano-' + str(step) +'.model'
		self._logger.info("writing model to " + str(file) )

		with open(file, 'wb') as f:
			cPickle.dump(self._out_vecs, f)
