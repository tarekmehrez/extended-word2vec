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


		# run MODE

		self._parser.add_argument(	'-f','--extract-feats', action='store_true', dest='feats',
									help='MODE: read in data')

		self._parser.add_argument(	'-t','--train', action='store', dest='train',
									help='MODE: train vector space (after reading in data)')

		self._parser.add_argument(	'-p','--plot', action='store', dest='plot',
									help='MODE: visualize vector space')

		# options for extracting feats
		self._parser.add_argument(	'-i','--input-dir', action='store', dest='dir',
									help='input directory containing text files, entities file & sources.csv')

		self._parser.add_argument(	'-d','--dim', action='store', dest='dim',
									help='vector dimensions, DEFUALT: 100',type=int,default=100)

		self._parser.add_argument(	'-w','--window-size', action='store', dest='cw',
									help='context window size, DEFUALT: 5',type=int,default=5)

		# optios for training
		self._parser.add_argument(	'-it','--iter', action='store', dest='iter',
									help='learning iterations, DEFUALT: 10',type=int,default=10)

		self._parser.add_argument(	'-n','--neg-samples', action='store', dest='neg',
									help='negative samples, DEFUALT: 5',type=int,default=5)

		self._parser.add_argument(	'-a','--alpha', action='store', dest='alpha',
									help='learning rate, DEFUALT: 0.01',type=float,default=0.01)

		self._parser.add_argument(	'-r','--reg', action='store', dest='reg',
									help='regularization term, DEFUALT: 0.01',type=float,default=0.01)

		# option for plotting
		self._parser.add_argument(	'-v','--version-number', action='store', dest='run',
									help='run number')

		self._parser.add_argument(	'-m','--model', action='store', dest='model',
									help='vector space to be visualized')




	def parse(self, args):
		self._logger.info("parsing arguments")
		results = self._parser.parse_args()

		if not (results.feats or results.train or results.plot):
			self._help_exit()

		if results.feats:
			if results.train or results.plot:
				self._logger.info("you can either read in a data, train a model or visualize one at a time")
				self._help_exit()


 			if not results.dir:
				self._logger.info("you have to specify the input dir to read the data")
				self._help_exit()

			return ('extract-feats', results.dir, results.cw, results.neg)

		if results.train:
			if results.feats or results.plot :
				self._logger.info("you can either read in a data, train a model or visualize one at a time")
				self._help_exit()

			if not os.path.exists('pickled/features.pkl'):
				self._logger.info("you have to read in the data first")
				self._help_exit()

			if not (results.train == 'gensim' or results.train == 'theano'):
				self._logger.info("you can only using gensim or theano")
				self._help_exit()

			return ('train', results.train, results.dim, results.iter, results.alpha, results.reg, results.run, results.model)

		if results.plot:
			if results.feats or results.train:
				self._logger.info("you can either read in a data, train a model or visualize one at a time")
				self._help_exit()

			if not os.path.exists(results.model):

				self._logger.info("the model you are trying to visualize does not exist")
				self._help_exit()

			if not (results.plot == 'show' or results.plot == 'save'):
				self._logger.info("you can only show or save figures. Save calculates euclidean distances as well")
				self._help_exit()

			return('plot', results.plot, results.model, results.run)




