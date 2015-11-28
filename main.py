import logging

from corpus import Corpus

entities = ['FIFA','USA','Iran','UK','Switzerland','Syria']

corpus = Corpus()
corpus.read_input('art-data', entities)
corpus.make_contexts(3)
corpus.write()