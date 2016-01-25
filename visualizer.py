import cPickle, sys
import gensim


import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA


class Visualizer:

	def __init__(self,logger, args):

		self._logger = logger
		self._mode ,self._model_path, self._run = args

		self._logger.info('reading model')

		if 'gensim' in self._model_path:
			model = gensim.models.Word2Vec()
			self._model = model.load(self._model_path)
			self._vocab = self._model.vocab

			self._src = 'gensim'

		else:
			self._model = self._load_model(self._model_path)
			self._src = 'theano'


		self._pca()

		self._vis()

		if self._mode == 'save' and self._src == 'theano':
			self._distances()



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

			if self._mode == 'save':
				plt.annotate(label, xy = (x, y),fontsize='xx-small')
			else:
				plt.annotate(label, xy = (x, y))

		if self._mode == 'save':
			file = 'fig-' + self._model_path.split('/')[-1] + '.eps'
			plt.savefig(file, format='eps', dpi=1200)
		else:
			plt.show()


	def _distances(self):

		entities = dict()

		for idx,i in enumerate(self._vocab):
			if i[0].isupper():
				entities[idx] = i

		distances = np.zeros((len(entities),len(entities)))

		red = self._reduced[entities.keys()]
		for idx,val in entities.iteritems():
			curr = self._reduced[idx]
			distances[idx] = np.sqrt(np.sum(np.power((curr - red),2),axis=1))



		file = 'distances.' + str(self._run) + '.txt'
		f = open(file,'w')
		f.write('\t')
		for i in entities:
			f.write(self._vocab[i][0:2] + self._vocab[i][-2:]+ '\t')
		f.write('\n')
		for i,row in enumerate(distances):
			f.write(self._vocab[i][0:2] + self._vocab[i][-2:] + '\t')

			for j,item in enumerate(row):
				f.write(str("%.2f" % distances[i,j])+ "\t")
			f.write('\n')
		f.close()

	def _load_model(self, path):

		with open(path, 'r') as f:
			self._model = cPickle.load(f)[0]

		with open('vocab.pkl', 'r') as f:
			self._vocab = cPickle.load(f)
		self._vocab = self._vocab.keys()

		return self._model.get_value()[:-1]
