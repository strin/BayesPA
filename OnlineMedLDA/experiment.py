from libbayespa import *
import math
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

m_K = 20

pamedlda = paMedLDAave({"num_topic"			:	m_K, 
						"batchsize"			:	1,
						"alpha"				:	0.5,
						"beta"				:	0.45,
						"c"					:	500,
						"l"					:	16,
						"I"					:	2,
						"J"					:	1,
						"sigma2"			:	1e-3,
						"train_file"		:	"../../data/20ng_train.gml",
						"test_file"			:	"../../data/20ng_test.gml",
						"epoch"				: 	1})

# test the prediction accuracy on 20 newsgroup.
def acc_test():
	pamedlda.train(11269)
	print pamedlda.infer(100)

# visualize topic dist.
def visualize_topic(category_i, num_period):
	dir_name = 'visualize_dist_paMedLDAave'
	try:
	    os.stat(dir_name)
	except:
	    os.mkdir(dir_name)
	num_iter = 11269
	num_category = 20
	label = pamedlda.labelOfInference()
	dist_all = list()
	for period in range(num_period):
		pamedlda.train(int(math.ceil(num_iter/num_period)))
		pamedlda.infer(100)
		mat = np.array(pamedlda.topicDistOfInference(category_i))
		count = np.array([0]*20)
		dist = np.zeros((num_category, m_K))
		for ni in range(len(label)):
			mat[ni] = mat[ni]/sum(mat[ni])
			dist[label[ni]] = dist[label[ni]]+mat[ni]
			count[label[ni]] = count[label[ni]]+1
		for ci in range(num_category):
			dist[ci] = dist[ci]/float(count[ci])
			plt.clf()
			plt.bar(range(m_K), dist[ci])
			plt.savefig('%s/%d_%d.eps'%(dir_name, period, ci))
		# print dist
		dist_all.append(dist[ci])
	sio.savemat('%s/visualize_topic'%(dir_name), {'dist':dist})

if __name__ == '__main__':
	visualize_topic(0, 10)
	# acc_test()



