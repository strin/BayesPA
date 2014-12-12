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

class BinaryclassTest(unittest.TestCase):
  def setUp(me):
    m_K = 5
    me.batchsize = 64
    config = {  "num_topic"      :  m_K, 
                "alpha"        :  0.5,
                "beta"        :  0.45,
                "c"          :  1,
                "l"          :  164,
                "I"          :  1,
                "J"          :  3,
                "train_file"    :  "../../../data/AtheismReligionMisc_Binary_train_nomalletstopwrd.gml",
                "test_file"      :  "../../../data/AtheismReligionMisc_Binary_test_nomalletstopwrd.gml",
                "dic_file"      :   "../../../data/dic.txt",
                "epoch"        :   1}
    (me.docs, me.labels, T1) = read_gml('../../../data/AtheismReligionMisc_Binary_train_nomalletstopwrd.gml')
    (me.test_docs, me.test_labels, T2) = read_gml('../../../data/AtheismReligionMisc_Binary_test_nomalletstopwrd.gml')
    me.T = max(T1, T2)
    config['#label'] = 2
    config['#word'] = me.T
    me.pamedlda = paMedLDAgibbs(config)

  def test_acc(me):
    allind = set(range(len(me.docs)))
    while len(allind) > 0:
      print len(allind)
      if len(allind) >= me.batchsize:
        ind = npr.choice(list(allind), me.batchsize, replace=False)
      else:
        ind = list(allind)
      allind -= set(ind)
      batch_doc = [me.docs[i] for i in ind]
      batch_label = [me.labels[i] for i in ind]
      me.pamedlda.train(batch_doc, batch_label)
    print 'infer'
    me.pamedlda.infer(me.test_docs, me.test_labels, 100)
    print 'test accuracy = ', me.pamedlda.testAcc()
    assert(me.pamedlda.testAcc() > 0.8)


class MulticlassTest(unittest.TestCase):
  def setUp(me):
    m_K = 20
    me.batchsize = 512
    config = {  "num_topic"      :  m_K, 
                "batchsize"      :  me.batchsize,
                "alpha"        :  0.5,
                "beta"        :  0.45,
                "c"          :  1,
                "l"          :  164,
                "I"          :  1,
                "J"          :  3,
                "train_file"    :  "../../../data/20ng_train.gml",
                "test_file"      :  "../../../data/20ng_test.gml",
                "dic_file"      :   "../../../data/dic.txt",
                "epoch"        :   1}
    (me.docs, me.labels, T1) = read_gml('../../../data/20ng_train.gml')
    (me.test_docs, me.test_labels, T2) = read_gml('../../../data/20ng_test.gml')
    me.T = max(T1, T2)
    config['#label'] = 20
    config['#word'] = me.T
    me.pamedlda = paMedLDAgibbs(config)

  def test_acc(me):
    allind = set(range(len(me.docs)))
    while len(allind) > 0:
      print len(allind)
      if len(allind) >= me.batchsize:
        ind = npr.choice(list(allind), me.batchsize, replace=False)
      else:
        ind = list(allind)
      allind -= set(ind)
      batch_doc = [me.docs[i] for i in ind]
      batch_label = [me.labels[i] for i in ind]
      me.pamedlda.train(batch_doc, batch_label)
    print 'infer'
    print me.pamedlda.infer(me.test_docs, me.test_labels, 100)
    print 'test accuracy = ', me.pamedlda.testAcc()
    assert(me.pamedlda.testAcc() > 0.79)

  
if __name__ == '__main__':
  unittest.main()



