from libbayespagibbs import *
import sys
import math
import os
import unittest
import medlda
import csv
from datetime import datetime

datadir = os.environ['datadir']
num_topic = int(os.environ['topic'])
num_pass = int(os.environ['pass'])
batchsize = int(os.environ['batchsize'])
stepsize = float(os.environ['stepsize'])

pamedlda = medlda.OnlineGibbsMedLDA(num_topic=num_topic, labels = 2,
                                    words = 61188, stepsize=stepsize)

for pi in range(num_pass):
    pamedlda.train_with_gml('%s/binary_train.gml' % datadir, batchsize)

(pred, ind, acc) = pamedlda.infer_with_gml('%s/binary_test.gml' % datadir, 100)

output = 'result/binary_large_topics.txt'
if not os.path.exists(os.path.dirname(output)):
    os.mkdir(os.path.dirname(output))

with open(output, 'a+') as f:
    f.write('topic %(num_topic)d pass %(num_pass)d batchsize %(batchsize)d datetime %(datetime)s stepsize %(stepsize)f acc %(acc)f\n' %
            dict(num_topic=num_topic, num_pass=num_pass, batchsize=batchsize,
                 datetime=datetime.now().strftime('%D+%T'), acc=acc,
                 stepsize=stepsize)
            )

