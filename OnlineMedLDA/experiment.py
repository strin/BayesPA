from libbayespa import *

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
print pamedlda.testAcc()