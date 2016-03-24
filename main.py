import argparse
import sys

from my_word2vec import MyWord2Vec

# input args
parser = argparse.ArgumentParser()

parser.add_argument('-data','--input_data',
					action='store',
					dest='data',
					help='Input Data')

parser.add_argument('-emb','--emb-size',
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

parser.add_argument('-e','--epochs',
					action='store',
					dest='epochs',
					help='number of epochs',
					type=int,
					default=5)

parser.add_argument('-r','--reg',
					action='store',
					dest='regularizer',
					help='Regularization Parameter',
					type=float,
					default=1.0)

parser.add_argument('-o','--output',
					action='store',
					dest='output',
					help='Output File',
					default='model-new.pkl')

parser.add_argument('-m','--model',
					action='store',
					dest='model',
					help='Existing Model')


args = parser.parse_args()

if not args.data:
	print 'input data must be specified'
	parser.print_help()
	sys.exit()

print args

model_instance = MyWord2Vec(args)
model_instance.read_data()
model_instance.train()

