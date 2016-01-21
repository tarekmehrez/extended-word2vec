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
		self._windows = []
		self._neg_samples = []
		self._freq = defaultdict(int)


	def read_freq(self):
		for file_path in self._files:
			curr = self._read_file(file_path)

			for token in curr:
				self._freq[token] += 1


		return self._freq

	def ctx_wid(self, global_vocab,cw):


		new_dict = self._my_entities
		for e in new_dict.keys():
			e_idx = global_vocab.index(e)

			self._my_entities[e_idx] = self._my_entities.pop(e)
			self._my_entities[e_idx].add_parallel_entity(e_idx)


		for file_path in self._files:

			curr = self._read_file(file_path)
			idx = map(lambda word: global_vocab.index(word), curr)
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


	def add_file(self,file_path):
		self._files.append(file_path)


	def get_files(self):
		return self._files

	def get_entities(self):
		return self._entities
