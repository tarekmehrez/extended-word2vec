import sys, logging

from arg_parser import ArgParser
from corpus import Corpus
from vector_space import VectorSpace

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)


parser = ArgParser(logger)
args = parser.parse(sys.argv)

train = args[0] == 'train'

if train:
	VectorSpace(logger,args[1:])
else:
	Corpus(logger,args[1:])