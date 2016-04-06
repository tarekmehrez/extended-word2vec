'''
Author: Tarek Mehrez
Desc:
- A simple script that takes in a text file with paths to artilces
- Applies NER using Stanford CoreNLP
- Appends name of source to each entity back to the file - so USA in afp becomes USA#AFP
- deletes the original text file, for memory issues
'''

import argparse
import os
import sys
import time
import nltk
import re

import numpy as np

from nltk.corpus import stopwords

from multiprocessing import Pool




def process(file):

	final_path = file + '.tok'

	with open(file, 'rb') as f:
		file_id, text = f.read().decode('utf-8').strip().split('\n',1)

	if '.com' in file_id:
		source = file_id.split('.com')[0].split('.')[-1]
	elif '.org' in file_id:
		source = file_id.split('.org')[0].split('.')[-1]
	elif '.de' in file_id:
		source = file_id.split('.de')[0].split('.')[-1]
	elif '.tw' in file_id:
		source = file_id.split('.tw')[0].split('.')[-1]
	else:
		source = '_'.join(file_id.split('_',2)[:2])

	# remove undesired chars
	text = re.sub('[^A-Za-z0-9\.,]+', ' ', text)

	# tokenize
	tokens = nltk.word_tokenize(text)

	all_text = ' '.join(tokens).split(' . ')

	with open(final_path, 'wb') as writer:
		writer.write(source + '\n')
		writer.write('\n'.join(all_text))




parser = argparse.ArgumentParser()

parser.add_argument('-i','--input',
					action='store',
					dest='input',
					help='File containing input files paths')


parser.add_argument('-p','--pools',
					action='store',
					dest='num_pools',
					help='Number of pools',
					type=int)


args = parser.parse_args()

if not ( args.input and args.num_pools):
	parser.print_help()
	sys.exit()

with open(args.input, 'rb') as f:
	files = f.read().strip().split('\n')

total = len(files)


pool = Pool(args.num_pools)
start = time.time()
print "Starting parallel processes for NER with number of files: %d" % total

for idx, x in enumerate(pool.imap_unordered(process, files)):
	print 'elapsed time for file %d out of %d : %d' %(idx, total,int(time.time() - start))
