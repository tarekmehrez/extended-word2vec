import gzip
import sys
import os
import argparse
import HTMLParser
import time

import numpy as np

from lxml import etree
from multiprocessing import Pool



def parse_file(file):

	global output_dir

	with gzip.open(file, 'rb') as f:
		file_content = np.array(f.read().decode('utf8').strip().split('</DOC>'))

	dir_name = file.split('/')[-1].split('.')[0]
	dir_name = output_dir + '/' + dir_name

	os.makedirs(dir_name)

	file_content= filter(None, file_content)

	splits = np.char.add(file_content, '</DOC>')

	print 'parsing %s at %s' % (file, time.ctime())
	for idx, doc in enumerate(splits):
		doc = doc.replace('&AMP;','')

		try:
			tree = etree.fromstring(doc)
		except:
			print 'ERROR OCCURRED in %s, doc no. %d' % (file, idx)
			continue

		doc_id =  tree.attrib['id']
		text = doc_id + '\n'

		if tree.xpath('.//HEADLINE'):
			text += tree.xpath('.//HEADLINE')[0].text
		text_tags = tree.xpath('.//TEXT')
		for text_tag in text_tags:

			if text_tag.text:
				text += text_tag.text

			p_tags = text_tag.xpath('.//P')

			for p in p_tags:
				text += p.text

		with open('%s/%s' % (dir_name, doc_id), 'wb') as f:
			f.write(text.encode('utf8'))




parser = argparse.ArgumentParser()


parser.add_argument('-paths','--paths',
					action='store',
					dest='paths',
					help='File containing input files paths')

parser.add_argument('-o','--output',
					action='store',
					dest='output',
					help='Output dir')

parser.add_argument('-p','--pools',
					action='store',
					dest='num_pools',
					help='Number of pools',
					type=int)


args = parser.parse_args()

if not ( args.paths and args.output and args.num_pools):
	parser.print_help()
	sys.exit()

with open(args.paths, 'rb') as f:
	files = f.read().strip().split('\n')

output_dir = args.output
total = len(files)
start = time.time()

pool = Pool(args.num_pools)
start = time.time()

print "Starting parallel processes with number of entries: %d" % total

for idx, x in enumerate(pool.imap(parse_file, files)):
	print 'elapsed time for file %d out of %d : %d' %(idx, total,int(time.time() - start))
