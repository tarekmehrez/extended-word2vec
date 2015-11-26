import os

import numpy as np
from collections import defaultdict

entities = ['FIFA','USA','Iran','UK','Switzerland','Syria']

vocab_idx = set()
sources_content = []


# read input, build vocab
for i, file in enumerate(os.listdir('art-data')):
	if file.startswith('mixture'):
		f = open('art-data/' + file, 'r')

		current = f.read().strip().split(' ')
		current = np.array(current)

		for e in entities:
			indices = np.where(current==e)
			current[indices] = e + '_' + str(i)

		vocab_idx.update(set(current))
		sources_content.append(current)


vocab_idx = dict(zip(dict.fromkeys(vocab_idx),range(len(vocab_idx))))
context = defaultdict(list)

# context window
cw = 3


# replace words by indices
for i, source in enumerate(sources_content):
	for word in vocab_idx:
		indices = np.where(source == word)
		sources_content[i][indices] = vocab_idx[word]
	sources_content[i] = sources_content[i].astype(np.int)

	# build context windows
	upper = lambda x: 0 if x < 0 else x
	for j,word in enumerate (sources_content[i]):
		curr = sources_content[i][upper(j-cw):j+cw+1]

		# delete current word from context list

		mask = np.ones(len(curr), np.bool)
		indices = np.where(curr==word)[0]
		mask[indices]=False
		curr = curr[mask]

		context[word] +=curr.tolist()






