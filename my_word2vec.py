
import argparse
import collections
import logging
import sys
import collections
import random
import os
import cPickle
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class MyWord2Vec:

	def __init__(self, args):

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self._logger = logging.getLogger(__name__)

		self._logger.info('Reading args...')

		self._args = args
		self._lr  = self._args.lr
		self.data_index = 0

		self._buildvocab()
		self._build_dataset()
		self._sample_dist()

	def set_batch_size(self):
		ctx, neg = self.generate_batch()
		self._context_tensor_size = len(ctx)
		self._sampled_tensor_size = len(neg)
		self.data_index = 0
		return self._context_tensor_size

	def _buildvocab(self):
		self._logger.info('Reading in data, creating vocab ...')
		with open(self._args.data, 'rb') as f:
			self._words = f.read().split(' ')

		self.vocab = list(set(self._words))
		self.vocab.append('UNK')
		self.vocab_size = len(self.vocab)

		self._logger.info('Number of tokens: %d' % len(self._words))
		self._logger.info('Vocab Size: %d' % len(self.vocab))

	def _build_dataset(self):
		self._logger.info('Building data set ...')

		data = np.array(self._words)
		vocab = self.vocab

		for idx, word in enumerate(self.vocab):
			data[data == word]= idx

		data = data.astype(int)
		dictionary= dict(zip(vocab, range(len(vocab))))
		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		count = dict(collections.Counter(self._words))
		count['UNK']=1

		unigrams_dict = dict()
		for word, value in count.iteritems():
			idx = dictionary[word]
			unigrams_dict[idx] = value

		unigrams = collections.OrderedDict(unigrams_dict)
		unigrams = np.array(unigrams.values(),dtype=float)

		self._unigrams = unigrams.tolist()

		self._data = data.tolist()
		self.data_size = len(self._data)
		self._dictionary = dictionary
		self._reverse_dictionary = reverse_dictionary
		self._count = count

		del(self._words)

	def _sample_dist(self):
		freq = np.power(self._unigrams / np.sum(self._unigrams), 0.75) # unigrams ^ 3/4
		self._dist = freq * (1 / np.sum(freq)) #normalize probabs
		del(self._unigrams)

	def _get_samples(self, size):
		samples = np.random.choice(range(self.vocab_size), size, p=self._dist)
		return samples

	def plot(self, title):
		tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
		low_dim_embs = tsne.fit_transform(self.out_embeddings.eval())


		labels = [obj._reverse_dictionary[key] for key in xrange(obj.vocab_size)]
		for label, x, y in zip(obj.vocab, low_dim_embs[:, 0], low_dim_embs[:, 1]):
			plt.plot(x,y,'x')
			plt.annotate(label, xy = (x, y),fontsize='xx-small')

		file = 'fig-%s.eps' % title
		plt.savefig(file, format='eps', dpi=1200)
		plt.clf()


	def build_graph(self):
		self._logger.info('Building tf graph...')

		self.graph = tf.Graph()

		with self.graph.as_default():

			init_width = 0.5 / self._args.emb_size

			# Shared variables holding input and output embeddings
			self.inp_embeddings = tf.Variable(	tf.random_uniform(
												[self.vocab_size, self._args.emb_size],
												-init_width, init_width))

			self.out_embeddings = tf.Variable(	tf.random_uniform(
												[self.vocab_size, self._args.emb_size],
												-init_width, init_width))

			# place holders, for batch inputs
			self.inp_ctx = tf.placeholder(	tf.int32,
											shape=(self._context_tensor_size))

			self.out_ctx = tf.placeholder(	tf.int32,
											shape=(self._context_tensor_size))

			self.inp_neg = tf.placeholder(	tf.int32,
											shape=(self._sampled_tensor_size))

			self.out_neg = tf.placeholder(	tf.int32,
											shape=(self._sampled_tensor_size))


			# embedding lookups to get vectors of specified indices (by placeholders)
			embed_inp_ctx = tf.nn.embedding_lookup(self.inp_embeddings, self.inp_ctx)
			embed_out_ctx = tf.nn.embedding_lookup(self.out_embeddings, self.out_ctx)

			embed_inp_neg = tf.nn.embedding_lookup(self.inp_embeddings, self.inp_neg)
			embed_out_neg = tf.nn.embedding_lookup(self.out_embeddings, self.out_neg)

			dot_ctx = tf.mul(embed_inp_ctx, embed_out_ctx)
			sum_ctx = tf.reduce_sum(dot_ctx, 1)
			ctx_expr = tf.log(tf.sigmoid(sum_ctx)) / self._context_tensor_size

			dot_neg = tf.mul(embed_inp_neg, embed_out_neg)
			sum_neg = tf.reduce_sum(dot_neg, 1)
			neg_expr = tf.log(tf.sigmoid(-sum_neg)) / self._sampled_tensor_size


			self.loss = (tf.reduce_sum(ctx_expr) + tf.reduce_sum(neg_expr))

			optimizer = tf.train.GradientDescentOptimizer(self._lr)
			self.train = optimizer.minimize(-self.loss,
											gate_gradients=optimizer.GATE_NONE)

	def lr_decay(self):
		decay_factor =  10.0 * (5.0 / float(self._args.epochs))
		lr = np.maximum(0.0001, self._lr / decay_factor)
		self._lr = round(lr,4)

	def generate_batch(self):

		context_words = []
		sampled_words = []

		# get current batch, curr_index: curr_index + batch_size
		current_data_batch = self._data[ self.data_index : self.data_index + self._args.batch_size ]
		self.data_index += (self._args.batch_size % self.data_size)

		# add extra UNKs for padding context windows
		padding_index = self._dictionary['UNK']
		lpadded = self._args.window // 2 * [padding_index] + current_data_batch + self._args.window // 2 * [padding_index]

		for idx, word in enumerate(current_data_batch):


			context = lpadded[idx:(idx + self._args.window)]
			samples = self._get_samples(self._args.samples)


			context_words += zip([word] * len(context), context)
			sampled_words += zip([word] * len(samples), samples)

		return context_words, sampled_words

# input args
parser = argparse.ArgumentParser()

parser.add_argument('-i','--input_data',
					action='store',
					dest='data',
					help='Input Data')

parser.add_argument('-e','--emb-size',
					action='store',
					dest='emb_size',
					help='Embeddings Size',
					type=int,
					default=100)

parser.add_argument('-w','--window_size',
					action='store',
					dest='window',
					help='Window Size',
					type=int,
					default=5)

parser.add_argument('-n','--neg_samples',
					action='store',
					dest='samples',
					help='Negative Samples',
					type=int,
					default=10)

parser.add_argument('-lr','--learning_rate',
					action='store',
					dest='lr',
					help='Learning Rate',
					type=float,
					default=1.0)

parser.add_argument('-b','--batch_size',
					action='store',
					dest='batch_size',
					help='Batch Size',
					type=int,
					default=50)

parser.add_argument('-epochs','--epochs',
					action='store',
					dest='epochs',
					help='number of epochs',
					type=int,
					default=7)

parser.add_argument('-o','--output',
					action='store',
					dest='output',
					help='Output File',
					default='model-new.pkl')

args = parser.parse_args()

if not args.data:
	print 'input data must be specified'
	parser.print_help()
	sys.exit()

print args

obj = MyWord2Vec(args)
single_batch = obj.set_batch_size()

obj.build_graph()

batch_iter = obj.data_size // single_batch
check_point = batch_iter / 4
epochs = args.epochs

if os.path.exists(args.output):
	with open(args.output, 'rb') as f:
		final_embeddings = cPickle.load(f)
else:
	print '[*] Starting training for %d epochs, each with %d number of batches' % (epochs, batch_iter)

	with tf.Session(graph=obj.graph) as sess:
		tf.initialize_all_variables().run()

		first_start = time.time()

		avg = 0
		for epoch in xrange(1, epochs+1):
			start = time.time()
			print '[*] training, epoch num: %d, out of %d with learning rate: %f' % (epoch, epochs, obj._lr)
			for batch in xrange(batch_iter):

				context_tuples, sampled_tuples = obj.generate_batch()

				inp_ctx, out_ctx = zip(*context_tuples)
				inp_neg, out_neg = zip(*sampled_tuples)

				feed_dict = {	obj.inp_ctx: inp_ctx,
								obj.out_ctx: out_ctx,
								obj.inp_neg: inp_neg,
								obj.out_neg: out_neg,	}

				try:
					result, _ = sess.run([obj.loss, obj.train], feed_dict=feed_dict)
				except:
					print '[*] encountered an error, ignoring'
					pass

				avg += result

				if batch % check_point == 0 and batch != 0:
					print '\t[*][*] batch %s out of %s, avg cost=%s, time so far: %ds' % (batch, batch_iter, avg/batch,int(time.time()-start))

			avg /= batch_iter

			print '[*] Done epoch %s out of %s, avg cost=%s, time taken: %ds ' % ( epoch,epochs,avg, int(time.time()-start))
			avg = 0
			obj.lr_decay()
			print '_______________________\n'

		print '[*] Total training time: %ds' % int(time.time()-first_start)
		final_embeddings = obj.out_embeddings.eval()

with open(args.output, 'wb') as f:
	cPickle.dump(final_embeddings, f, protocol=cPickle.HIGHEST_PROTOCOL)

pca = PCA(n_components=2)
pca.fit(final_embeddings)
low_dim_embs = pca.transform(final_embeddings)

obj.vocab.remove('UNK')

labels = [obj._reverse_dictionary[key] for key in xrange(obj.vocab_size-1)]
for label, x, y in zip(obj.vocab, low_dim_embs[:, 0], low_dim_embs[:, 1]):
	plt.plot(x,y,'x')
	plt.annotate(label, xy = (x, y))

plt.show()
plt.clf()