import sys

from arg_parser import ArgParser
from corpus import Corpus
# from vec_space import VectorSpace

parser = ArgParser()
results = parser.parse(sys.argv)

print results
train = results[0] == 'train'

if not train:
	Corpus(results[1:])
