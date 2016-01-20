import numpy as np
import gensim

class GensimModel:

	def __init__(self,logger, corpus):
		self._logger = logger

		self._logger.info('initializing theano model')
		self._corpus = corpus


	def train(self):
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

		self._model = gensim.models.Word2Vec()
		self._model.build_vocab(sentences)

		self._model.train(sentences)
		self._logger.info('done training vectors with gensim')
		self._save_model()


	def _save_model(self):

		file = 'gensim.model'
		self._logger.info('writing gensim model to ' + str(file))
		self._model.save(file)
