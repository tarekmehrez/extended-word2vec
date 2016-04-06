import cPickle
import logging
import random
import time
import sys
import os
import math

from data_reader import DataReader

from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class MyWord2Vec:

	def __init__(self, args):

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self._logger = logging.getLogger(__name__)

		self._logger.info('Initializing Model...')
		self._logger.info('Reading Args...')

		self._args = args
		self._lr  = self._args.lr
		self._data_index = 0

		self._context_tensor_size = 0
		self._sampled_tensor_size = 0


	def read_data(self):

		self._logger.info('Reading meta data...')

		self._reader = DataReader(self._logger)

		(self._vocab,
		self._vocab_size,
		self._dictionary,
		self._reverse_dictionary,
		self._unigrams,
		self._arts_srcs,
		self._srcs_ents,
		self._ents_srcs) = self._reader.read_meta_files(self._args.data)


		with open(self._args.output + '-labels-dict.pkl', 'wb') as f:
			cPickle.dump(self._reverse_dictionary, f,protocol=cPickle.HIGHEST_PROTOCOL)

		with open(self._args.output + '-vocab-dict.pkl', 'wb') as f:
			cPickle.dump(self._dictionary, f,protocol=cPickle.HIGHEST_PROTOCOL)

		self._number_of_srcs = len(set(self._srcs_ents.keys()))

		self._sample_dist()

	def _load_model(self, file_path):
		with open(file_path, 'rb') as f:
			embeddings = cPickle.load(f)
		return embeddings

	def _save_model(self, file_path, embeddings):
		with open(file_path, 'wb') as f:
			cPickle.dump(embeddings, f, protocol=cPickle.HIGHEST_PROTOCOL)

	def _sample_dist(self):
		freq = np.power(self._unigrams / np.sum(self._unigrams), 0.75) # unigrams ^ 3/4
		self._dist = freq * (1 / np.sum(freq)) #normalize probabs

	def _get_samples(self, size):
		samples = np.random.choice(range(self._vocab_size), size, p=self._dist)
		return samples

	def _plot(self,title,embeddings):

		self._logger.debug('Plotting...')

		pca = PCA(n_components=2)
		pca.fit(embeddings)
		low_dim_embs = pca.transform(embeddings)
		labels = [self._reverse_dictionary[key] for key in xrange(self._vocab_size)]


		for label, x, y in zip(labels, low_dim_embs[:, 0], low_dim_embs[:, 1]):
			plt.plot(x,y,'x')

			if title != 'final':
				plt.annotate(label, xy = (x, y),fontsize='xx-small')
			else:
				plt.annotate(label, xy = (x, y))


		if title is 'final':
			plt.show()
		else:
			file = 'fig-%s.eps' % title
			plt.savefig(file, format='eps', dpi=1200)

		plt.clf()


	def _build_graph(self):

		self._logger.info('Building tf graph...')

		self.graph = tf.Graph()
		with self.graph.as_default():

			self.make_vars()
			self.build_expr()
			self.optimize()

	def make_vars(self):
		init_width = 0.5 / self._args.emb_size

		# Shared variables holding input and output embeddings
		self.inp_embeddings = tf.Variable(	tf.random_uniform(
											[self._vocab_size, self._args.emb_size],
											-init_width, init_width))

		self.out_embeddings = tf.Variable(	tf.random_uniform(
											[self._vocab_size, self._args.emb_size],
											-init_width, init_width))


	def build_expr(self):

		self.inp_ctx = tf.placeholder(tf.int32,shape=(None))
		self.out_ctx = tf.placeholder(tf.int32,shape=(None))

		self.inp_neg = tf.placeholder(tf.int32,shape=(None))
		self.out_neg = tf.placeholder(tf.int32,shape=(None))

		self.out_ents = tf.placeholder(tf.int32,shape=(None))
		self.other_ents = tf.placeholder(tf.int32,shape=(None))

		ctx_batch_size = tf.shape(self.inp_ctx)[0]
		neg_batch_size = tf.shape(self.out_ctx)[0]
		ents_constant = tf.shape(self.out_ents)[0]

		src_constnt = tf.constant(self._number_of_srcs,dtype=tf.float32)

		# embedding lookups to get vectors of specified indices (by placeholders)
		embed_inp_ctx = tf.nn.embedding_lookup(self.inp_embeddings, self.inp_ctx)
		embed_out_ctx = tf.nn.embedding_lookup(self.out_embeddings, self.out_ctx)

		embed_inp_neg = tf.nn.embedding_lookup(self.inp_embeddings, self.inp_neg)
		embed_out_neg = tf.nn.embedding_lookup(self.out_embeddings, self.out_neg)

		embed_entities = tf.nn.embedding_lookup(self.out_embeddings, self.out_ents)
		embed_other_entities = tf.nn.embedding_lookup(self.out_embeddings, self.other_ents)

		dot_ctx = tf.mul(embed_inp_ctx, embed_out_ctx)
		sum_ctx = tf.reduce_sum(dot_ctx, 1)
		ctx_expr = tf.log(tf.sigmoid(sum_ctx)) / tf.cast(ctx_batch_size, tf.float32)

		dot_neg = tf.mul(embed_inp_neg, embed_out_neg)
		sum_neg = tf.reduce_sum(dot_neg, 1)
		neg_expr = tf.log(tf.sigmoid(-sum_neg)) / tf.cast(neg_batch_size, tf.float32)


		avg_ents = tf.div(tf.reduce_sum(embed_other_entities, 1), src_constnt)
		ents_diff = tf.square(tf.sub(embed_entities, avg_ents))
		reg_expr =  self._args.regularizer * tf.reduce_sum(ents_diff) / tf.cast(ents_constant, tf.float32)


		self.loss = tf.reduce_sum(ctx_expr) + tf.reduce_sum(neg_expr) - reg_expr

	def optimize(self):
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self.train = optimizer.minimize(-self.loss,
										gate_gradients=optimizer.GATE_NONE)



	def lr_decay(self):
		decay_factor =  10.0 * (5.0 / float(self._args.epochs))
		lr = np.maximum(0.0001, self._lr / decay_factor)
		self._lr = round(lr,4)

	def _ents_matrices(self):

		self._logger.info('Preparing named entites for this source')

		# get political entities
		source_entities = np.array(self._srcs_ents[self._current_source])


		corresponding_ents = list()
		padding_index = self._dictionary['UNK']
		# get corresponding entities and replace tokens by ids
		for ent in source_entities:

			base_ent = ent.split('_',-1)[0]
			temp = np.array(self._ents_srcs[base_ent])
			'''
			TODO:
			Remve entity from its correspondings list
			'''
			for curr_ent in temp:
				temp[temp==curr_ent] = self._dictionary[curr_ent]

			temp = temp.astype(int).tolist()
			temp += [padding_index] * (self._number_of_srcs - len(temp))
			corresponding_ents.append(temp)


		# replace entities' tokens by ids
		source_ents_ids = source_entities
		for ent in source_ents_ids:
			source_ents_ids[source_ents_ids==ent] = self._dictionary[ent]

		source_ents_ids = source_ents_ids.astype(int)

		self._current_entities = source_ents_ids
		self._corresponding_ents = corresponding_ents


	def generate_batch(self):

		context_words = []
		sampled_words = []

		# get current batch, curr_index: curr_index + batch_size
		current_data_batch = self._data[ self._data_index : self._data_index + self._args.batch_size ]
		self._data_index += (self._args.batch_size % self._data_size)

		# add extra UNKs for padding context windows
		padding_index = self._dictionary['UNK']
		lpadded = self._args.window // 2 * [padding_index] + current_data_batch + self._args.window // 2 * [padding_index]

		for idx, word in enumerate(current_data_batch):


			context = lpadded[idx:(idx + self._args.window)]
			samples = self._get_samples(self._args.samples)

			context_words += zip([word] * len(context), context)
			sampled_words += zip([word] * len(samples), samples)

		inp_ctx, out_ctx = zip(*context_words)
		inp_neg, out_neg = zip(*sampled_words)

		feed_dict = {	self.out_ents: self._current_entities,
						self.other_ents: self._corresponding_ents,
						self.inp_ctx: inp_ctx,
						self.out_ctx: out_ctx,
						self.inp_neg: inp_neg,
						self.out_neg: out_neg,	}

		return feed_dict


	def _prepare_file(self, file_path):
		data = np.array(self._reader.read_file(file_path, self._dictionary))

		self._data = data.astype(int).tolist()

		self._data_size = len(data)



	def train(self):

		if os.path.exists(self._args.output):
			embeddings = self._load_model(self._args.output)
			self._plot('final',embeddings)
			return

		self._build_graph()
		self._logger.info('Starting training ...')

		with tf.Session(graph=self.graph) as sess:

			tf.initialize_all_variables().run()
			first_start = time.time()
			start = time.time()

			for epoch in xrange(1, self._args.epochs+1):

				self._logger.info(	'[*] training, epoch num: %d, out of %d with learning rate: %f' \
									% (epoch, self._args.epochs, self._lr))

				total_batches = 0
				batches_so_far = 0

				avg = 0

				for file_path in self._arts_srcs:

					self._current_source = self._arts_srcs[file_path]
					self._ents_matrices()


					self._logger.info('Reading file %s' % file_path)
					self._prepare_file(file_path)

					file_batches = self._data_size / self._args.batch_size
					check_point = file_batches / 4
					total_batches += file_batches

					for batch in xrange(file_batches):
						batches_so_far += 1
						feed_dict = self.generate_batch()
						cost, _ = sess.run([self.loss, self.train], feed_dict=feed_dict)


						# if math.isnan(cost) or math.isinf(cost):
						# 	self._logger.info('[*] Encountered NaN or Inf, stopping training')
						# 	final_embeddings = prev_emb.eval()
						# 	break

						avg += cost

						# if batch % check_point == 0 and batch != 0:
						self._logger.info(	'\t[*][*] batch %s out of %s, avg cost=%s, time so far: %ds' \
											% (batch, file_batches, avg/batches_so_far,int(time.time()-start)))

					self._data_index = 0
					self._logger.info(	'[*] Done file %s, avg cost=%s, time taken: %ds ' \
										% ( file_path,avg/file_batches, int(time.time()-start)))

				avg /= total_batches
				self._logger.info(	'[*] Done epoch %s out of %s, avg cost=%s, time taken: %ds ' \
									% ( epoch,self._args.epochs,avg, int(time.time()-start)))

				avg = 0
				self.lr_decay()
				print '________________________________________________\n'

			self._logger.info('[*] Total training time: %ds' % int(time.time()-first_start))
			final_embeddings = self.out_embeddings.eval()
		self._save_model(self._args.output, final_embeddings)
		self._plot('final',final_embeddings)
