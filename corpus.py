import os, csv, sys, cPickle

import numpy as np

from collections import defaultdict
from collections import OrderedDict
from scipy import stats


from entity import Entity
from source import Source

# reads in corpus, must have files, sources.csv and entities.txt

class Corpus:

	def __init__(self,logger, args):

		self._logger = logger

		if os.path.exists('corpus.pkl'):
			self._logger.info('corpus.pkl already exists')
			sys.exit(1)

		self._args = args
		self._dir, self._cw, self._samples = args

		self._read_dir()
		self._create_vocab()


		self._create_ctx_windows()
		self._create_neg_samples()
		self._create_ent_maps()
		self._save()


	def _read_dir(self):

		self._logger.info('reading source files & named entities...')


		with open(self._dir + '/entities.txt','r') as f:
			reader=f.read()

		entities = reader.split('\n')[:-1]

		self._entities = {name: Entity(name) for name in entities}

		with open(self._dir + '/sources.csv','r') as f:
			reader = csv.reader(f)
			files = [files for files in reader]

		self._source_files = files
		self._data_sources = dict()

		for file_path,source_name in self._source_files:

			new_source = Source(source_name,self._entities)
			new_source.add_file(self._dir + '/' + file_path)

			self._data_sources[source_name] = new_source


	def _create_vocab(self):

		self._logger.info('creating vocab')

		self._freq = defaultdict(int)


		for src, obj in self._data_sources.iteritems():
			for key, val in obj.read_freq().iteritems():
				self._freq[key] += val

		# TODO handle periods

		self._freq = OrderedDict(sorted(self._freq.items()))
		self._freq['<PADDING_TOKEN>']=0
		self._logger.info('corpus size: ' + str(sum(self._freq.values())))
		self._logger.info('vocab size: ' + str(len(self._freq)))

	def _create_ctx_windows(self):

		self._logger.info('creating ctx windows')

		vocab = self._freq.keys()

		for src, obj in self._data_sources.iteritems():
			obj.ctx_wid(vocab,self._cw)


	def _create_neg_samples(self):

		self._logger.info('creating neg samples')


		idx = range(len(self._freq))
		freq = np.array(self._freq.values(), dtype=np.float)
		freq = np.power(freq / np.sum(freq), 0.75) # unigrams ^ 3/4
		dist = freq * (1 / np.sum(freq)) #normalize probabs

		for src, obj in self._data_sources.iteritems():
			obj.neg_sam(self._samples,idx, dist, self._cw)

	def _create_ent_maps(self):

		self._logger.info('creating entities indices')
		for src, obj in self._data_sources.iteritems():

			obj.set_p_entities(len(self._data_sources))

	def _save(self):
		self._logger.info("saving corpus object to corpus.pkl")
		self._logger = None

		with open('corpus.pkl', 'wb') as f:
			cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)

		with open('vocab.pkl', 'wb') as f:
			cPickle.dump(self._freq, f, protocol=cPickle.HIGHEST_PROTOCOL)



	def get_vocab(self):
		return self._freq.keys()

	def get_sources(self):
		return self._data_sources

	def get_entities(self):
		return self._entities