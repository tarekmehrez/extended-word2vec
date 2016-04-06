'''
Author: Tarek Mehrez
Desc:
- Takes in a pickle file of vectors to be visualized alongside the vocab in a dictionary
- Displays the visualized vector space using matplotlib
- Could also save the plot to external eps images
'''

import cPickle
import argparse
import sys
import pickle

import tsne

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

def plot(title,embeddings, labels,use_tsne):

	if not use_tsne:
		low_dim_embs = PCA(n_components=2).fit_transform(embeddings)
	else:
		low_dim_embs = tsne.tsne(embeddings,2,len(embeddings), 50.0)

	if title:
		for label, x, y in zip(labels, low_dim_embs[:, 0], low_dim_embs[:, 1]):
			plt.plot(x,y,'x')
			plt.annotate(label, xy = (x, y),fontsize='xx-small')

		file = 'fig-%s.eps' % title
		plt.savefig(file, format='eps', dpi=1200)

	plt.clf()
	for label, x, y in zip(labels, low_dim_embs[:, 0], low_dim_embs[:, 1]):
		plt.plot(x,y,'x')
		plt.annotate(label, xy = (x, y))

	plt.show()
	plt.clf()

if  __name__ =='__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('-v','--vectors',
						action='store',
						dest='vectors',
						help='Pkl file  with input vectors to be visualized')

	parser.add_argument('-l','--labels',
						action='store',
						dest='labels',
						help='Pkl file with dictionary containing vocab and their indices')

	parser.add_argument('-o','--output',
						action='store',
						dest='output',
						help='output file to save plot as image')


	parser.add_argument('-tsne','--tsne',
						action='store_true',
						dest='use_tsne',
						default=False,
						help='plot with t-SNE')

	args = parser.parse_args()


	# TODO replace by argparse positional arguments
	if not (args.vectors and args.labels):
		print 'missing args'
		parser.print_help()
		sys.exit()

	print args

	with open(args.vectors, 'rb') as f:
		embeddings = cPickle.load(f)

	with open(args.labels, 'rb') as f:
		reverse_dictionary = cPickle.load(f)

	labels = []
	for key in xrange(len(reverse_dictionary)):
		if reverse_dictionary[key] != 'UNK':
			labels.append(reverse_dictionary[key])

	print 'plotting...'
	plot(args.output, embeddings, labels,args.use_tsne)
