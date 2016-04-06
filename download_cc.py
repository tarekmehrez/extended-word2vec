import StringIO
import gzip
import os
import requests
import sys
import time
import argparse
import warc

import eatiht.v2 as v2
from multiprocessing import Pool

reload(sys)
sys.setdefaultencoding('utf8')

def download_page(record):

	offset, length = int(record['offset']), int(record['length'])
	offset_end = offset + length - 1

	prefix = 'https://aws-publicdatasets.s3.amazonaws.com/'
	resp = requests.get(prefix + record['filename'], headers={'Range': 'bytes={}-{}'.format(offset, offset_end)})

	raw_data = StringIO.StringIO(resp.content)
	f = gzip.GzipFile(fileobj=raw_data)

	data = f.read()
	return data

def clean_warc(input):

	text = v2.extract(input)
	warc_content = warc.WARCFile(fileobj=StringIO.StringIO(input))

	for record in warc_content:
		url,date =  record['WARC-Target-URI'], record['WARC-Date']

	return '%s,%s\n%s' % (url, date, text)

def run(item):
	global output


	record = eval(item)

	# if record['mime'] != 'text/html':
	# 	return

	try:
		warc_data = download_page(record)
		text = clean_warc(warc_data)
		file_name = record['url'].replace('https://','').replace('/','_')[:60]
		with open('%s/%s.text' % (output, file_name), 'wb') as f:
			f.write(text)
	except:
		print 'encountered error in %s, ignoring' % record['url']

parser = argparse.ArgumentParser()
parser.add_argument('-p','--pools', action='store', dest='num_pools', help='Number of pools', type=int)
parser.add_argument('-i','--input', action='store', dest='input', help='Input file with commoncrawl records')
parser.add_argument('-o','--output', action='store', dest='output', help='Output directory to write files')


args = parser.parse_args()

if not (args.num_pools and args.input):
	print 'missing args'
	parser.print_help()
	sys.exit()

if not os.path.exists(args.output):
	print args.output + ' doesnt exist, creating it...'
	os.makedirs(args.output)

output = args.output

print 'reading records'
with open(args.input, 'rb') as f:
	records = f.read().split('\n')

print "starting download"



pool = Pool(args.num_pools)
start = time.time()
total = len(records)
print "Starting parallel processes with number of entries: %d" % total

for idx, x in enumerate(pool.imap_unordered(run, records)):
	print 'elapsed time for file %d out of %d : %d' %(idx, total,int(time.time() - start))

