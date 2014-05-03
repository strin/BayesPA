from libbayespa import *
import scipy.io as sio

pamedlda = paMedLDAave({"num_topic"			:	20, 
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
pamedlda.train(11269)
pamedlda.infer(100)
topic = [None]*20
for ci in range(20):
	topic[ci] = pamedlda.topicMatrix(ci)
sio.savemat('topic.mat', {'topic':topic})