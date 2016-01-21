import sys
import numpy as np
from collections import defaultdict

class Source:

	def __init__(self, name, glob_ents):

		self._name = name

		self._files = []
		self._global_entities = glob_ents
		self._my_entities = dict()

		self._tokens = []

		self._content = []
		self._windows = []
		self._neg_samples = []
		self._freq = defaultdict(int)


	def read_freq(self):
		for file_path in self._files:
			curr = self._read_file(file_path)

			for token in curr:
				self._freq[token] += 1

			self._content.append(curr)


		return self._freq

	def ctx_wid(self, global_vocab,cw):


		new_dict = dict()

		for e in self._my_entities:
			e_idx = global_vocab.index(e)
			new_dict[e_idx] = self._my_entities[e]
			new_dict[e_idx].add_parallel_entity(e_idx)

		self._my_entities = new_dict
		for file_content in self._content:

			idx = map(lambda word: global_vocab.index(word), file_content)
			self._tokens += idx
			self._windows += self._context_win(idx,cw)

		self._tokens = np.array(self._tokens, dtype=np.int32)
		self._windows = np.array(self._windows, dtype=np.int32)


	def _read_file(self, in_file):

		# TODO handle punctuations, stop, frequent and rare words
		file_path = in_file

		with open(file_path, 'r') as f:
			curr_file = f.read().strip()

		curr_file = np.array(curr_file.split(' '), dtype='|S30')


		for e, obj in self._global_entities.iteritems():
			if e in curr_file:


				new = e + '_' + self._name
				curr_file[curr_file==e] = new

				obj.add_source(self._name)
				self._my_entities[new] = obj

		return curr_file

	def _context_win(self,input,cw):

		lpadded = cw // 2 * [-1] + input + cw // 2 * [-1]
		windows = [lpadded[i:(i + cw)] for i in range(len(input))]
		return windows



	def neg_sam(self,samples,idx, dist,cw):

		self._neg_samples = np.zeros((len(self._windows), samples + cw), dtype=np.int32)

		for example in range(len(self._neg_samples)):
			self._neg_samples[example] = np.random.choice(idx, (samples + cw), p=dist)


	def shuffle_data(self):
		data = np.hstack((self._windows, self._neg_samples))
		data = np.column_stack((self._tokens, data))

		np.random.shuffle(data)
		self._tokens, self._windows, self._neg_samples = np.split(data, [1, self._windows.shape[1] + 1], axis=1)
		self._tokens = np.asarray(self._tokens).reshape(-1)


	def set_p_entities(self, total_sources):
		self._e_prime = []
		for e, obj in self._my_entities.iteritems():
			curr = obj.get_parallel_entities()

			curr += [-1] * (total_sources - len(curr))
			self._e_prime.append(curr)

		self._e_prime = np.array(self._e_prime, dtype=np.int32)


	def get_windows(self):
		return self._windows

	def get_neg_samples(self):
		return self._neg_samples

	def get_tokens(self):
		return self._tokens

	def add_file(self,file_path):
		self._files.append(file_path)

	def get_entities(self):
		return np.array(self._my_entities.keys(), dtype=np.int32)

	def get_p_entities(self):
		return self._e_prime

	def get_files(self):
		return self._files

