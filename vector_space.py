import theano
import logging
import pickle

from theano import tensor as T
from collections import defaultdict


class VectorSpace:

	def __init__(self, args):

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self._logger = logging.getLogger(__name__)
		self._logger.info('initializing vector space')
		self._load_corpus()

		print args


	def _load_corpus(self):
		self._logger.info("loading corpus object from corpus.pkl")

		with open('corpus.pkl', 'r') as f:
			self._corpus = pickle.load(f)