BayesPA
=======

This repository contains code for the following paper:

* Shi, Tianlin, and Jun Zhu. "[Online Bayesian Passive-Aggressive Learning](http://ml.cs.tsinghua.edu.cn/~tianlin/research/bayespa/BayesPA_full.pdf)." Proceedings of The 31st International
Conference on Machine Learning. 2014.

Specifically, the code is a <b> streaming implementation of MedLDA </b>

<h2> What is MedLDA ? </h2>
<p> Maximum Entropy Discriminant LDA (MedLDA) is a max-margin supervised topic model. It <i>jointly trains Latent Dirichlet
Allocation (LDA) with SVM</i>, and obtains a topic representation more suitable for discriminative tasks such as
classification. </p>

<h2> What is BayesPA ? </h2>
Online Bayesian PA (BayesPA) is a generalization of classic Passive-Aggressive learning to the Bayesian and latent-variable setting. For every incoming mini-batch of documents, BayesPA first applies Bayes' rule to update the LDA topic model, then projects the posterior distribution to a region where the hinge-loss on the mini-batch data is minimized. 

How to Use
==========

The python interface of Online MedLDA is simple. 

To use, simply 

```
import medlda
```

To create a classifer with 2 labels and 61188 words, 

```
pamedlda = medlda.OnlineGibbsMedLDA(num_topic = 5, labels = 2, words = 61188)
```

The training and inference are also straightfoward, 

```
pamedlda.train_with_gml('../data/binary_train.gml', batchsize=64)
(pred, ind, acc) = pamedlda.infer_with_gml('../data/binary_test.gml', num_sample=100)
```

Please refer to [docs](medlda.html) for more detals.

Installing Online MedLDA 
========================

This release is for early adopters of this premature software. Please let us know if you have comments or suggestions. Contact: tianlinshi [AT] gmail.com 

Online MedLDA is written in C++ 11, with a friendly python interface. It depends on gcc >= 4.8, python (numpy  >= 1.7.0, distutils) and boost::python.
To install, follow the instructions below.

Dependencies (Ubuntu)
---------------------
```
# system dependency
sudo apt-get install libboost-all-dev gcc-4.8
sudo apt-get install python-numpy
```

Dependencies (OS X, Homebrew)
---------------------
```
brew install gcc
brew install boost --cc=gcc-4.9
brew install boost-python --cc=gcc-4.9
pip install numpy scipy
```

Installation
---------------------
```
sudo python setup.py install
```

Citation
========

If you use online MedLDA in your work, please cite


<i>Shi, T., & Zhu, J. (2014). Online Bayesian Passive-Aggressive Learning. In Proceedings of The 31st International Conference on Machine Learning (pp. 378-386).</i>




License (GPL V3)
================

Copyright (C) 2014 Tianlin Shi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributeCd in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


