import argparse
import sys
import re
import cPickle

from collections import defaultdict

def run(data):
	xml = data.split('-->')[1]
	countries = xml.split('<Country>')
	countries = countries[1:]

	global locations

	for country in countries:
		r = re.compile('<CountryCode>(.*?)</CountryCode>')
		m = r.search(country)
		if m:
			code = m.group(1).encode('utf-8')

		r = re.compile('<CountryName>(.*?)</CountryName>')
		m = r.search(country).group(1)
		tokens = [i.lower() for i in m.split('_')]
		try:
			locations[' '.join(tokens).encode('utf-8')] = code
		except UnicodeDecodeError:
				continue

		r = re.compile('<COW-Alpha>(.*?)</COW-Alpha>')
		m = r.search(country).group(1)
		try:
			locations[m.lower().encode('utf-8')] = code
		except UnicodeDecodeError:
				continue
		r = re.compile('<ISO3166-alpha3>(.*?)</ISO3166-alpha3>')
		m = r.search(country).group(1)
		try:
			locations[m.lower().encode('utf-8')] = code
		except UnicodeDecodeError:
			continue

		# nationalities
		get_nested(country,'<Nationality>','</Nationality>',code)

		# capitals
		get_nested(country,'<Capital>','</Capital>',code)
		get_nested(country,'<MajorCities>','</MajorCities>',code)
		get_nested(country,'<GeogFeatures>','</GeogFeatures>',code)


def get_nested(country,tag,end_tag, code):

	global locations

	splits = country.split(tag)
	if len(splits) < 2:
		return
	l = splits[1].split(end_tag)[0].split('\n')
	l = filter(None, l)

	if not l:
		return

	for inside in l:


		splits = inside.strip().split()
		if not splits:
			return

		big_token = splits[0]
		if '{' in big_token:
			r = re.compile('{(.*?)}')
			big_token = r.search(big_token).group(1).strip()

		tokens = big_token.split('_')
		tokens = filter(None, tokens)
		tokens = [i.lower() for i in tokens]

		if len(tokens) > 1:
			try:
				locations[' '.join(tokens).encode('utf-8')] = code
			except UnicodeDecodeError:
				continue
		else:
			try:
				locations[tokens[0].lower().encode('utf-8')] = code
			except UnicodeDecodeError:
				continue

parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', action='store', dest='input', help='path to CountryInfo.txt')
args = parser.parse_args()

if not args.input:
	print 'missing args'
	parser.print_help()
	sys.exit()


with open(args.input, 'rb') as f:
	data= f.read().strip()

locations = dict()

run(data)

for i in locations:
	print i, locations[i]



with open('country_info.pkl', 'wb') as f:
	cPickle.dump(locations, f, protocol=cPickle.HIGHEST_PROTOCOL)


