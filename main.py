import sys, logging

from arg_parser import ArgParser
from corpus import Corpus
from vector_space import VectorSpace
from visualizer import Visualizer



logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)


parser = ArgParser(logger)
args = parser.parse(sys.argv)


if args[0] == 'train':
	VectorSpace(logger,args[1:])
elif args[0] == 'corpus':
	Corpus(logger,args[1:])
else:
	Visualizer(logger, args[1:])