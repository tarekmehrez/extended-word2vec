import sys

from arg_parser import ArgParser
from corpus import Corpus
from vector_space import VectorSpace

parser = ArgParser()
results = parser.parse(sys.argv)

train = results[0] == 'train'

if train:
	VectorSpace(results[1:])
else:
	Corpus(results[1:])