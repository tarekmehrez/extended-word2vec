class Entity:

	def __init__(self, name):

		self._name = name
		self._sources = []
		self._parallel_entities = []

	def set_idx(self, idx):
		self._idx = idx

	def add_source(self, src):
		self._sources.append(src)

	def add_parallel_entity(self, e):
		self._parallel_entities.append(e)

	def get_parallel_entities(self):
		return self._parallel_entities