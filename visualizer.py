import cPickle, sys
import gensim


import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA


class Visualizer:

	def __init__(self,logger, args):

		self._logger = logger
		model_path = args[0]

		self._logger.info('reading model')

		if 'gensim' in model_path:
			model = gensim.models.Word2Vec()
			self._model = model.load(model_path)
			self._vocab = self._model.vocab

			self._src = 'gensim'

		elif 'mikolov' in model_path:
			self._model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=False)  # C text format
			self._vocab = self._model.vocab

			self._src = 'gensim'

		else:
			self._model = self._load_model(model_path)
			self._src = 'theano'


		self._pca()
		self._vis()



	def _pca(self):
		self._logger.info('applying pca to vectors')


		if self._src != 'theano':
			self._vectors= []
			for token_vector in self._vocab:
				self._vectors.append(self._model[token_vector])
			self._vectors = np.array(self._vectors)
		else:
			self._vectors = self._model
		pca = PCA(n_components=2)
		pca.fit(self._vectors)
		self._reduced = pca.transform(self._vectors)



	def _vis(self):

		self._logger.info('visualizing vectors')


		for label, x, y in zip(self._vocab, self._reduced[:, 0], self._reduced[:, 1]):
			plt.plot(x,y,'x')
			plt.annotate(label, xy = (x, y))

		plt.show()

	def _load_model(self, path):

		with open(path, 'r') as f:
			self._model = cPickle.load(f)

		with open('vocab.pkl', 'r') as f:
			self._vocab = cPickle.load(f)
		self._vocab = self._vocab.keys()

		return self._model.get_value()[:-1]