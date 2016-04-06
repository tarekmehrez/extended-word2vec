import cPickle
import os
import sys
import argparse
import time
import re

from multiprocessing import Pool

def run(files):


	global locations
	global output

	total = len(files)

	for idx, file in enumerate(files):

		out = file + ".res"
		print 'file no. %d out of %d' % (idx, total)

		with open(file, 'rb') as f:
			text = f.read().strip().decode('utf8').lower()

		expr = re.compile('#(.*?)\s')
		source = expr.search(text).group(1)
		if not source:
			continue

		text = text.replace('#' + source, '')

		for i in locations:
			text = text.replace(' %s ' % i, ' %s ' % locations[i] + '#' + source)
			text = text.replace(' %s' % i, ' %s' % locations[i] + '#' + source)
			text = text.replace('%s ' % i, '%s ' % locations[i] + '#' + source)

		with open(out, 'wb') as f:
			f.write(text)



parser = argparse.ArgumentParser()


parser.add_argument('-i','--input',
					action='store',
					dest='input',
					help='File containing input files paths')

parser.add_argument('-o','--output',
					action='store',
					dest='output',
					help='Output path')

parser.add_argument('-d','--dict',
					action='store',
					dest='cc_dict',
					help='Path to CountryInfo dict, produced by cc_to_dict.py')

parser.add_argument('-p','--pools',
					action='store',
					dest='num_pools',
					help='Number of pools',
					type=int)

args = parser.parse_args()

if not ( args.input and args.cc_dict and args.num_pools):
	print 'missing args'
	parser.print_help()
	sys.exit()


with open(args.input, 'rb') as f:
	input_content = f.read().strip().split('\n')

with open(args.cc_dict, 'rb') as f:
	locations = cPickle.load(f)

pool = Pool(args.num_pools)
start = time.time()
total =len(input_content)
print "Starting parallel processes for NER with number of files: %d" % total

for i in input_content:
	run(input_content)

# for idx, x in enumerate(pool.imap_unordered(run, input_content)):
# 	print 'elapsed time for file %d out of %d : %d' %(idx, total,int(time.time() - start))


