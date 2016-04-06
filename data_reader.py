import collections

import numpy as np
import nltk

class DataReader:

	def __init__(self, logger):
		self._logger = logger


	def read_meta_files(self, data_dir):

		# vocab ~> frequencies
		# build id dictionaries, and unigrams (for negative sampling)
		with open('%s/meta/vocab.txt' % data_dir) as f:
			vocab_freqs = np.array([line.strip().split(',') for line in f], dtype=np.str)

		count = dict(zip(vocab_freqs[:,0], vocab_freqs[:,1]))



		vocab = count.keys()
		vocab_size = len(vocab)
		dictionary= dict(zip(vocab, range(vocab_size)))
		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

		self._logger.info('Vocab Size: %d' % vocab_size)

		unigrams_dict = dict()
		for word, value in count.iteritems():
			idx = dictionary[word]
			unigrams_dict[idx] = value

		unigrams = collections.OrderedDict(unigrams_dict)
		unigrams = np.array(unigrams.values(),dtype=float)
		unigrams = unigrams.tolist()

		# source ~> list of all articles paths of this source
		with open('%s/meta/article_src.csv' % data_dir) as f:
			arts_srcs_pairs = np.array([line.strip().split(',') for line in f], dtype=np.str)
		arts_srcs = dict(zip(arts_srcs_pairs[:,0],arts_srcs_pairs[:,1]))

		self._logger.info('Number of sources: %d' % len((set(arts_srcs.values()))))
		self._logger.info('Number of articles: %d' % len(arts_srcs))

		# source ~> all entities occurring in this source
		with open('%s/meta/src_ents.csv' % data_dir) as f:
			src_ents = np.array([line.strip().split(',') for line in f])

		src_ents_dict = dict()
		for entry in src_ents:
			src_ents_dict[entry[0]] = entry[1:]

		srcs_ents = src_ents_dict

		# entity ~> all sources where this entity occur
		with open('meta/ents_srcs.csv') as f:
			ents_srcs_pairs = np.array([line.strip().split(',') for line in f])

		ents_srcs = dict()
		for entry in ents_srcs_pairs:
			ents_srcs[entry[0]] = entry[1:]

		self._logger.info('Number of entities: %d' % len(ents_srcs))

		entities = [len(ents_srcs[key]) for key in ents_srcs]
		total_ents = sum(entities)

		self._logger.info('Number of source-specific entities: %d' % total_ents)


		return (vocab,
				vocab_size,
				dictionary,
				reverse_dictionary,
				unigrams,
				arts_srcs,
				srcs_ents,
				ents_srcs)


	def read_file(self, file, dictionary):

		words = self._preprocess(file)
		words_ids = np.array(words)

		for word in dictionary:
			words_ids[words_ids == word] = dictionary[word]


		return words_ids.tolist()

	def _preprocess(self, file):


		with open(file, 'rb') as f:
			return f.read().strip().split(' ')


