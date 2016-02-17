import sys, logging

from arg_parser import ArgParser
from features import Features
from vector_space import VectorSpace
from visualizer import Visualizer



logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)


parser = ArgParser(logger)
args = parser.parse(sys.argv)


print args
if args[0] == 'train':
	VectorSpace(logger,args[1:])

elif args[0] == 'extract-feats':
	Features(logger,args[1:])

else:
	Visualizer(logger, args[1:])