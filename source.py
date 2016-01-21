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

		return self._tokens, self._windows

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





	def set_p_entities(self, total_sources):
		self._e_prime = []
		for e, obj in self._my_entities.iteritems():
			curr = obj.get_parallel_entities()

			curr += [-1] * (total_sources - len(curr))
			self._e_prime.append(curr)

		return self._my_entities.keys(), self._e_prime

	def add_file(self,file_path):
		self._files.append(file_path)


