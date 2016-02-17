import cPickle, sys

from gensim_model import GensimModel
from theano_model import TheanoModel

class VectorSpace:

	def __init__(self, logger, args):

		self._logger = logger
		features = self._load_features()


		self._logger.info('initializing vector space')

		self._args = args
		train_option = args[0]


		if train_option == 'gensim':
			self._train_gensim(features)
		# else:
			# self._train_theano(features)


	def _train_gensim(self, features):
		model = GensimModel(self._logger, features)
		model.train()

	# def _train_theano(self,features):

	# 	model = TheanoModel(self._logger,features.get_vocab(), self._args)
	# 	model.compile()
	# 	model.train(self._corpus.get_sources())


	def _load_features(self):
		self._logger.info("loading features object from features.pkl")

		with open('pickled/features.pkl', 'r') as f:
			features = cPickle.load(f)

		return features


