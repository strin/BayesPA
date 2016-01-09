from libbayespagibbs import *
import sys
import math
import os
import unittest
import medlda

class BinaryclassTest(unittest.TestCase):
  def test_acc(me):
    pamedlda = medlda.OnlineGibbsMedLDA(num_topic = 5, labels = 2, words = 61188)
    pamedlda.train_with_gml('../data/binary_train.gml', 64)
    (pred, ind, acc) = pamedlda.infer_with_gml('../data/binary_test.gml', 10)
    print 'pred = ', zip(ind, pred)
    print 'acc = ', acc
    assert(acc > 0.80)

  def test_acc_K80(me):
    pamedlda = medlda.OnlineGibbsMedLDA(num_topic = 80, labels = 2, words = 61188, stepsize=25)
    pamedlda.train_with_gml('../data/binary_train.gml', 32)
    (pred, ind, acc) = pamedlda.infer_with_gml('../data/binary_test.gml', 10)
    print 'pred = ', zip(ind, pred)
    print 'acc = ', acc
    assert(acc > 0.79)


class MultiLabelTest(unittest.TestCase):
    def test_f1(me):
        pamedlda = medlda.OnlineMultitaskGibbsMedLDA(num_topic = 5, labels = 20, words = 61188)
        pamedlda.train_with_gml('../data/wiki_train_subset.gml', 64)
        (pred, ind, acc) = pamedlda.infer_with_gml('../data/wiki_test.gml', 10)
        print 'pred = ', zip(ind, pred)
        print 'acc = ', acc
        assert(acc > 0.40)


if __name__ == '__main__':
  unittest.main()



