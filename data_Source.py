import nmupy as np

class DataSource:

	def __init__(self, name):
		self.name = name

	def add_articles(self, articles):
		self.articles_path = articles

	def add_entities(self, articles):
		self.entities = entities

	def ents_to_ids(self, dictionary):

		ents = np.array(self.entities)
		for i in ents:
			ents[ents == i] == dictionary[i]
		self.entities = ents.tolist()