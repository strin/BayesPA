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
    (pred, ind, acc) = pamedlda.infer_with_gml('../data/binary_test.gml', 100)
    print 'pred = ', zip(ind, pred)
    print 'acc = ', acc
    assert(acc > 0.80)
  
if __name__ == '__main__':
  unittest.main()



