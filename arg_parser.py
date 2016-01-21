import argparse, sys, os

class ArgParser:

	def __init__(self, logger):

		self._logger = logger


		self._parser = argparse.ArgumentParser()
		self._add_args()

	def _help_exit(self):
		self._parser.print_help()
		sys.exit(1)

	def _add_args(self):


		self._parser.add_argument(	'--corpus', action='store', dest='corpus',
									help='MODE: read in corpus')

		self._parser.add_argument(	'--train', action='store', dest='train',
									help='MODE: train vector space (after reading in corpus)')

		self._parser.add_argument(	'--plot', action='store', dest='plot',
									help='MODE: visualize vector space')


		self._parser.add_argument(	'--dir', action='store', dest='dir',
									help='input directory containing text files, entities file & sources.csv')

		self._parser.add_argument(	'--dim', action='store', dest='dim',
									help='vector dimensions, DEFUALT: 100',type=int,default=100)

		self._parser.add_argument(	'--cw', action='store', dest='cw',
									help='context window size, DEFUALT: 5',type=int,default=5)

		self._parser.add_argument(	'--iter', action='store', dest='iter',
									help='learning iterations, DEFUALT: 10',type=int,default=10)

		self._parser.add_argument(	'--neg', action='store', dest='neg',
									help='negative samples, DEFUALT: 5',type=int,default=5)


		self._parser.add_argument(	'--batch', action='store', dest='batch',
									help='batch size, DEFUALT: 25',type=float,default=25)

		self._parser.add_argument(	'--alpha', action='store', dest='alpha',
									help='learning rate, DEFUALT: 0.01',type=float,default=0.01)

		self._parser.add_argument(	'--reg', action='store', dest='reg',
									help='regularization term, DEFUALT: 0.01',type=float,default=0.01)

		self._parser.add_argument(	'--run', action='store', dest='run',
									help='run number')


		self._parser.add_argument(	'--model', action='store', dest='model',
									help='vector space to be visualized')




	def parse(self, args):
		self._logger.info("parsing arguments")
		results = self._parser.parse_args()

		if not (results.corpus or results.train or results.plot):
			self._help_exit()

		if results.corpus:
			if results.train or results.plot:
				self._logger.info("you can either read in a corpus, train a model or visualize one at a time")
				self._help_exit()


 			if not results.dir:
				self._logger.info("you have to specify the input dir to read the corpus")
				self._help_exit()

			return ('corpus', results.dir, results.cw, results.neg)

		if results.train:
			if results.corpus or results.plot :
				self._logger.info("you can either read in a corpus, train a model or visualize one at a time")
				self._help_exit()

			if not os.path.exists('corpus.pkl'):
				self._logger.info("you have to read in the corpus first")
				self._help_exit()

			if not (results.train == 'gensim' or results.train == 'theano'):
				self._logger.info("you can only using gensim or theano")
				self._help_exit()

			return ('train', results.train, results.dim, results.iter, results.batch, results.alpha, results.reg, results.run)

		if results.plot:
			if results.corpus or results.train:
				self._logger.info("you can either read in a corpus, train a model or visualize one at a time")
				self._help_exit()

			if not os.path.exists(results.model):

				self._logger.info("the model you are trying to visualize does not exist")
				self._help_exit()

			return('plot', results.model)




