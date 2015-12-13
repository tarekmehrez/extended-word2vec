import logging
import os

from corpus import Corpus
# from vector_space import VectorSpace


logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)

'''
TODO command line args for:

entities file
input_dir with articles
pickle files to be loaded
context window
alpha - learning rate
learning iterations
regularization parameter

'''

# TODO read entities from file
entities = ['FIFA','USA','Iran','UK','Switzerland','Syria']

# files to be read
# TODO clean this
files = ['vocab.pkl','freq.pkl','ctx_words.pkl']

# directory with all articles
input_dir = 'art-data'

corpus = Corpus(input_dir, entities)

# vec_space = VectorSpace(corpus)