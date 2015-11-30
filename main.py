from corpus import Corpus
from vector_space import VectorSpace

entities = ['FIFA','USA','Iran','UK','Switzerland','Syria']

corpus = Corpus()
# corpus.read_input('art-data', entities)
# corpus.make_contexts(3)
# corpus.write()

vec_space = VectorSpace(corpus.read())