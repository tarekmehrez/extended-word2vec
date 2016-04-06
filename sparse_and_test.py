'''
Takes in embeddings and vocab dictionary:
1- Identifies named entities from vocab
2- Extracts their ids and embeddings
3- Calculate deltas for each entity
4- Match deltas to features in testing data
5- Apply sparse Coding
6- Train on Logistic Regression with SGD (scikit)

'''


import argparse
import os
import sys
import cPickle
import time


from sklearn import linear_model

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import sparse_encode

from sklearn.metrics import f1_score, accuracy_score


import tensorflow as tf
import numpy as np


def extract_features(first_way):

	global args


	print 'reading data'
	with open(args.dict, 'rb') as f:
		dictionary = cPickle.load(f)

	with open(args.vecs, 'rb') as f:
		embeddings = np.array(cPickle.load(f))

	with open(args.meta +'/ents_sources.csv') as f:
		ents_srcs_pairs = np.array([line.strip().split(',') for line in f])

	ents_srcs = dict()
	for entry in ents_srcs_pairs:
		ents_srcs[entry[0]] = entry[1:]


	# source ~> all entities occurring in this source
	with open(args.meta +'/src_ents.csv') as f:
		src_ents = np.array([line.strip().split(',') for line in f])

	srcs_ents = dict()
	for entry in src_ents:
		srcs_ents[entry[0]] = entry[1:]


	emb_size = embeddings.shape[1]
	ents_matrix = np.ndarray(shape=(len(ents_srcs),emb_size))

	if first_way:
		print 'going for: how are others perceiving the entity'

		for idx, base_ent in enumerate(ents_srcs):

			expr = np.zeros(shape=emb_size)

			for ent_src in ents_srcs[base_ent]:
				others = ents_srcs[base_ent]
				curr_ent = dictionary[ent_src]

				curr_corres = []
				for i in others:
					curr_corres.append(dictionary[i])


				expr += np.abs(embeddings[curr_ent] - np.average(embeddings[curr_corres],axis=0))

			ents_matrix[idx] = expr
	else:
		print 'going for: how is the entity perceiving others'

		for idx, base_ent in enumerate(ents_srcs):
			expr = np.zeros(shape=emb_size)
			for_this_entity = ents_srcs[base_ent]

			for src_based_ent in for_this_entity:
				curr_idx = dictionary[src_based_ent]
				base_src_ent = src_based_ent.split('_',-1)[0]

				if base_src_ent in srcs_ents:
					parallel_ents = srcs_ents[base_src_ent]
					other_idxs = []

					for parallel_ent in parallel_ents:
						other_idxs.append(dictionary[parallel_ent])

					expr += np.abs(embeddings[curr_idx] - np.average(embeddings[other_idxs],axis=0))
			ents_matrix[idx] = expr

	print ents_matrix

	with open(args.input, 'rb') as f:
		training_file = f.read().split('\n')

	tuples = []
	labels = []

	print 'reading training file'
	for line in training_file:
		pairs = line.split()
		tuples.append(pairs[:2])
		labels.append(pairs[2])

	print 'deltas to features'

	base_ents = ents_srcs.keys()
	features = np.ndarray(shape=(len(tuples),emb_size))
	for idx, pair in enumerate(tuples):

		first = base_ents.index(pair[0])
		second = base_ents.index(pair[1])
		features[idx] = np.square(ents_matrix[first] - ents_matrix[second])

	print 'sparse coding'
	print features
	sparse_vectors = to_sparse(features,args.dim)
	return sparse_vectors, labels

def to_sparse(X,dim):

	sparse_dict = MiniBatchDictionaryLearning(dim)
	sparse_dict.fit(X)
	sparse_vectors = sparse_encode(X, sparse_dict.components_)

	for i in sparse_vectors:
		print i

	return sparse_vectors

def train(X,Y,cv_val):


	X = np.array(X)
	Y = np.array(Y)

	total_acc = 0.0
	total_f = 0.0

	print 'starting tranining with %d-fold cv ' % cv_val

	for i in range(cv_val):

		print 'number %d' % i
		X_splits = np.array_split(X,cv_val)
		Y_splits = np.array_split(Y,cv_val)

		X_test = X_splits[i]
		Y_test = Y_splits[i]

		del(X_splits[i])
		del(Y_splits[i])

		X_train = np.concatenate(X_splits)
		Y_train = np.concatenate(Y_splits)

		classifier = linear_model.SGDClassifier(n_iter=1000,loss='log')
		classifier.fit(X_train, Y_train)

		preds = classifier.predict(X_test)
		curr_f1 = f1_score(Y_test, preds, average='micro')
		curr_acc = accuracy_score(Y_test, preds)

		total_acc += curr_acc
		total_f += curr_f1
		print curr_f1
		print curr_acc


	print 'final fscore: %f' % (total_f / cv_val)
	print 'final accuracy: %f' % (total_acc /cv_val)

parser = argparse.ArgumentParser()

parser.add_argument('-i','--input_file',
					action='store',
					dest='input',
					help='input file with tuples to be trained')

parser.add_argument('-vecs','--input_vectors',
					action='store',
					dest='vecs',
					help='input embeddings')


parser.add_argument('-dict','--dictionary',
					action='store',
					dest='dict',
					help='vocab as dictionary')


parser.add_argument('-m','--meta',
					action='store',
					dest='meta',
					help='directory with meta model')


parser.add_argument('-sd','--sparse_dim',
					action='store',
					dest='dim',
					help='sparse vecs dims',
					type=int,
					default=30)


args = parser.parse_args()

if not (args.input or args.vecs or args.dict or args.meta):
	print 'missing args'
	parser.print_help()
	sys.exit()

print args

# calculate deltas
features, labels = extract_features(False)

# train with new feature representation
train(features, labels,20)
