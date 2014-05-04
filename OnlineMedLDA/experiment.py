from libbayespa import *
import math
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

m_K = 20
config = {				"num_topic"			:	m_K, 
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
						"dic_file"			: 	"../../data/dic.txt",
						"epoch"				: 	1}
pamedlda = paMedLDAave(config)

# test the prediction accuracy on 20 newsgroup.
def acc_test():
	pamedlda.train(11269)
	print pamedlda.infer(100)

# visualize topic dist.
def visualize_topic(category_i):
	dir_name = 'visualize_dist_paMedLDAave_%d'%(category_i)
	try:
	    os.stat(dir_name)
	except:
	    os.mkdir(dir_name)
	dic = file(config['dic_file']).readlines()
	num_iter = 11269
	num_category = 20
	periods = [1,16,256,4096,11269]
	label = pamedlda.labelOfInference()
	dist_all = list()
	topwords_all = list()
	for period in periods:
		print 'period = ', period
		pamedlda.train(int(period))
		pamedlda.infer(100)
		print 'test acc = ', pamedlda.testAcc()
		mat = np.array(pamedlda.topicDistOfInference(category_i))
		topwords = np.array(pamedlda.topWords(category_i, 10))
		def ind2words(topwords, dic):
			topwords_list = list()
			for i in range(len(topwords)):
				row = list()
				for j in range(len(topwords[0])):
					row.append(dic[topwords[i][j]].replace('\n', ''))
				topwords_list.append(row)
			return topwords_list
		topwords = ind2words(topwords, dic)
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
			plt.savefig('%s/%d_%d.eps'%(dir_name, ci, period))
		# print dist
		dist_all.append(dist[ci])
		topwords_all.append(topwords)
	sio.savemat('%s/visualize_topic'%(dir_name), {'dist':dist_all})
	topwords_output = open('%s/topwords.txt'%(dir_name), 'w')
	for topwords in topwords_all:
		for row in topwords:
			topwords_output.write(' & '.join(row)+'\n')
		topwords_output.write('\n\n')
	topwords_output.close()

if __name__ == '__main__':
	visualize_topic(1)
	# acc_test()



