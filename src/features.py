import os, csv, sys, cPickle

import numpy as np

from collections import defaultdict
from collections import OrderedDict
from scipy import stats


# reads in corpus, must have files, sources.csv and entities.txt

class Features:


	def __init__(self,logger, args):

		self._logger = logger

		if os.path.exists('corpus.pkl'):
			self._logger.info('corpus.pkl already exists')
			sys.exit(1)

		data_dir, window_size, sample_size = args

		self._read_data(data_dir,window_size)
		self._word_to_idx()
		self._sample(sample_size)

		self._save()


	def _read_data(self, data_dir, window_size):

		self._logger.info('reading data...')

		# read in csv files with source names and file paths
		with open(data_dir + '/data_sources.csv','rb') as f:
			data_files = csv.reader(f)

			input_tokens = []
			target_tokens = []
			freq = defaultdict(int)
			data_sources = []
			# read in content of data files
			for file_path, data_source in data_files:

				data_sources.append(data_source)
				# to do, handle periods, punctuations and line breaks
				with open(data_dir+ '/' + file_path, 'rb') as f:

					curr_tokens = f.read().split(' ')
					curr_tokens = np.char.add(curr_tokens, '_' + data_source).tolist()

					# count frequencies and create context windows
					for token in curr_tokens:
						freq[token] += 1

					target_tokens += self._get_windows(curr_tokens, window_size)
					input_tokens += curr_tokens

		freq['<PADDING>']=0


		self.input_tokens = np.array(input_tokens)
		self.target_tokens = np.array(target_tokens)

		# delete occurrence of input word in context window
		split = window_size / 2
		self.target_tokens = np.hstack((self.target_tokens[:,:split],self.target_tokens[:,split+1:]))

		self.freq = OrderedDict(freq)
		self.data_sources = data_sources

	def _get_windows(self,input,window_size):

		lpadded = window_size // 2 * ['<PADDING>'] + input + window_size // 2 * ['<PADDING>']
		windows = [lpadded[i:(i + window_size)] for i in range(len(input))]

		return windows

	def _word_to_idx(self):

		self._logger.info('replacing words by indices...')


		# replace words by their indices
		vocab = self.freq.keys()
		for idx, word in enumerate(vocab):

			self.input_tokens[self.input_tokens == word] = idx
			self.target_tokens[self.target_tokens == word] = idx

		self.input_tokens = np.array(self.input_tokens,dtype=np.int32)
		self.target_tokens = np.array(self.target_tokens,dtype=np.int32)


	def _sample(self,sample_size):

		# create negative samples, according to freq distribution
		self._logger.info('creating negative samples...')

		idx = range(len(self.freq))
		freq = np.array(self.freq.values(), dtype=np.float)
		freq = np.power(freq / np.sum(freq), 0.75) # unigrams ^ 3/4
		dist = freq * (1 / np.sum(freq)) #normalize probabs


		neg_samples = []

		for example in range(len(self.input_tokens)):
			neg_samples.append(np.random.choice(idx, sample_size, p=dist))

		self.neg_samples = neg_samples


	def _save(self):
		self._logger.info("saving features...")
		self._logger = None

		with open('pickled/features.pkl', 'wb') as f:
			cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)