 from collections import defaultdict

f = open('mixture3.txt','r')

s = f.read()
tokens = s.split(' ')

dict = defaultdict(float)

for token in tokens:
	dict[token] += 1

for i in dict:
	print i, dict[i]
