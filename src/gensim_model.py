import gensim

import numpy as np

class GensimModel:

	def __init__(self,logger, features):
		self._logger = logger
		self._features = features
		self._logger.info('initializing theano model')

		self._get_token_strings()




	def _get_token_strings(self):

		vocab = self._features.freq.keys()
		tokens = self._features.input_tokens

		tokens = tokens.astype(str)

		print tokens.dtype

		for idx, word in enumerate(vocab):
			tokens[tokens == idx] = word

		self.tokens = tokens

	def train(self):
		self._logger.info('starting training vectors with gensim')

		tokens = self.tokens

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

		self._model = gensim.models.Word2Vec()
		self._model.build_vocab(sentences)

		self._model.train(sentences)
		self._logger.info('done training vectors with gensim')
		self._save_model()


	def _save_model(self):

		file = 'models/gensim.model'
		self._logger.info('writing gensim model to ' + str(file))
		self._model.save(file)
