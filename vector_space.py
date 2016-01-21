import cPickle, sys

from gensim_model import GensimModel
from theano_model import TheanoModel

class VectorSpace:

	def __init__(self, logger, args):

		self._logger = logger
		self._load_corpus()


		self._logger.info('initializing vector space')

		self._args = args
		train_option = args[0]


		if train_option == 'gensim':
			self._train_gensim()
		else:
			self._train_theano()


	def _train_gensim(self):
		model = GensimModel(self._logger, self._corpus)
		model.train()

	def _train_theano(self):

		model = TheanoModel(self._logger,self._corpus.get_vocab(), self._args)
		model.compile()
		model.train(self._corpus.get_data())


	def _load_corpus(self):
		self._logger.info("loading corpus object from corpus.pkl")

		with open('corpus.pkl', 'r') as f:
			self._corpus = cPickle.load(f)



