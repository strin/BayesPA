import sys
sys.path.append('../')
from libbayespagibbs import *
import math
import scipy.io as sio
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import os
import unittest

def read_gml(path):
  lines = file(path).readlines()
  labels = []
  docs = []
  T = -1
  for line in lines[1:]:
    line = line.replace('\n', '')
    line = line.split(' ')
    label = int(line[1])
    doc = [int(token) for token in line[2:]]
    T = max(T, max(doc))
    labels += [label]
    docs += [doc]
  return (docs, labels, T)


m_K = int(os.environ['K'])
batchsize = 512
config = {  "#topic"      :  m_K,
            "batchsize"      :  batchsize,
            "train_file"    :  "../../../data/20ng_train.gml",
            "test_file"      :  "../../../data/20ng_test.gml",
            "dic_file"      :   "../../../data/dic.txt",
            "epoch"        :   1}
(docs, labels, T1) = read_gml('../../../data/20ng_train.gml')
(test_docs, test_labels, T2) = read_gml('../../../data/20ng_test.gml')
T = max(T1, T2)
print 'T', T
config['#label'] = 20
config['#word'] = T

pamedlda = paMedLDAgibbs(config)

allind = set(range(len(docs)))
print '> train'
while len(allind) > 0:
    print len(allind)
    if len(allind) >= batchsize:
        ind = npr.choice(list(allind), batchsize, replace=False)
    else:
        ind = list(allind)
    allind -= set(ind)
    batch_doc = [docs[i] for i in ind]
    batch_label = [labels[i] for i in ind]
    pamedlda.train(batch_doc, batch_label)

print '> infer'
pamedlda.infer(test_docs, test_labels, 100)
print 'test accuracy = ', pamedlda.testAcc()





