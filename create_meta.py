'''
Author: Tarek Mehrez
Desc:
- Create meta directory, required by MyWord2Vec class which contains the following files
-- articles' paths and their source names
-- Base enities and their occurrences in all sources
-- Each source and entities occurring in this exact source
-- Vocab and word frequencies
'''


import argparse
import os
import sys
import re

import numpy as np

from collections import defaultdict

def create(files):

	global srcs_ents
	global ents_srcs
	global vocab
	global articles_src_writer

	total = len(files)


	for idx, file in enumerate(files):


		print 'file no. %d out of %d' % (idx, total)

		with open(file, 'rb') as f:
			content = f.read().strip().decode('utf8')

		expr = re.compile('_(.*?)\s')
		match = expr.search(content)

		if not match:
			continue

		curr_src = match.group(1)

		for line in content.split('\n'):

			tokens = line.strip().split(' ')
			tokens = filter(None, tokens)
			for token in tokens:
				token = token
				vocab[token] += 1
				if '_' in token and len(token.split('_')) > 1:
					pair = token.split('_')
					current_src = pair[1]

					base_ent = pair[0]
					ents_srcs[base_ent].append(token)
					srcs_ents[current_src].append(token)

		articles_src_writer.write(','.join([file, current_src]) + '\n')


parser = argparse.ArgumentParser()

parser.add_argument('-i','--input',
					action='store',
					dest='input',
					help='File containing input files paths, or dir with existing meta files to be merged [combined with --merge]')

parser.add_argument('-o','--output',
					action='store',
					dest='output',
					help='Output path to save meta dir')

parser.add_argument('-m','--merge',
					action='store_true',
					dest='merge',
					default=False,
					help='Whether to merge meta dirs in input file')

args = parser.parse_args()

if not ( args.input and args.output):
	parser.print_help()
	sys.exit()

if not os.path.exists(args.output):
	os.makedirs(args.output)

articles_src_writer = open(args.output + '/article_src.csv', 'wb')

srcs_ents = defaultdict(list)
ents_srcs = defaultdict(list)
vocab = defaultdict(float)

with open(args.input, 'rb') as f:
	input_content = f.read().strip().split('\n')

create(input_content)


vocab['UNK'] = 1

print 'writing ents_srcs'
with open(args.output + '/ents_sources.csv', 'wb') as f:
	for key in ents_srcs:
		f.write(key + ',' + ','.join(list(set(ents_srcs[key]))) + '\n')

print 'writing srcs_ents'
with open(args.output + '/src_ents.csv', 'wb') as f:
	for key in srcs_ents:
		f.write(key + ',' + ','.join(list(set(srcs_ents[key]))) + '\n')

print 'writing vocab'
with open(args.output + '/vocab.txt', 'wb') as f:
	for key in vocab:
		f.write('%s,%s' % (key, vocab[key]) + '\n')

articles_src_writer.close()