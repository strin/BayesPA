from libbayespagibbs import *
import numpy.random as npr

def read_gml(path):
  """
    read from .gml file, which has the following format:
      line 0:     N         ;  N is the number of documents.
      line 1...N: M L w_1 w_2 w_3 ... w_M ; L is the label of the document, w_1, ..., w_M are words.

    Parameters
    ______________
    path:
      path of the .gml file.

    Returns
    _____________
    (docs, labels, word_set, label_set)

    docs: list
      documents, each doc is a list of words.
    labels: list of int
      the corresponding labels for the docs.
    word_set: set
      set of words in the document
    label_set: set
      set of labels in the document
  """
  lines = file(path).readlines()
  labels = []
  docs = []
  word_set = set()
  label_set = set()
  for line in lines[1:]:
    line = line.replace('\n', '')
    line = line.split(' ')
    label = int(line[1])
    doc = [int(token) for token in line[2:]]
    map(lambda w : word_set.add(w), doc)
    label_set.add(label)
    labels += [label]
    docs += [doc]
  return (docs, labels, word_set, label_set)

class OnlineGibbsMedLDA:
  def __init__(me, num_topic, labels, words, alpha = 0.5, beta = 0.45, c = 1, l = 164, v = 1,
              I = 1, J = 3, stepsize = 1):
    """
    initialize the online Gibbs MedLDA model.

    Parameters
    ______________
    num_topic: int
      the number of topics in LDA.
    labels: list of object
      the set of potential labels for the documents.
      if given an integer L, then it assumes the labels are [0 ... L-1]
    words: list of object
      the set of potential words in the documents.
      if given an integer T, then it assumes the words are [0 ... T-1]

    Optional Parameters
    _____________
    I: int, default = 1
      number of mean-field rounds for each BayesPA update.
    J: int, default = 3
      number of Gibbs samples in the mean-field update of latent variables (substract J_burnin)
    alpha: double, default = 0.5
      prior of document topic distribution.
    beta: double, default = 0.45
      prior of dictionary.
    c: double, default = 1
      regularization parameter of hinge-loss.
    l: double, default = 164,
      margin parameter of hinge-loss.
    v: double, default = 1
      prior weight \sim N(0, v^2).
    stepsize: double, default = 1
      weight for every data point.

    Returns
    _____________
    the OnlineGibbsMedLDA object.

    Examples
    _____________
    >>> medlda = OnlineGibbsMedLDA(5, [0, 1], ["athesim", "god", "beings"])
    >>> medlda = OnlineGibbsMedLDA(5, 2, 3)

    """
    me.labels = dict()
    if isinstance(labels, int):
      labels = range(labels)
    if isinstance(labels, list):
      map(lambda k, v : me.labels.update({k: v}), labels, range(len(labels)))
    else:
      raise TypeError("unrecognized labels type")

    me.words = dict()
    if isinstance(words, int):
      words = range(words)
    if isinstance(words, list):
      map(lambda k, v : me.words.update({k: v}), words, range(len(words)))

    print labels
    print len(me.words)

    config = {"#topic" : num_topic,
              "#label" : len(me.labels),
              "#word"  : len(me.words),
              "alpha"  : alpha,
              "beta"   : beta,
              "c"      : c,
              "v"      : v,
              "l"      : l,
              "I"      : I,
              "J"      : J,
              "stepsize": stepsize
              }

    me.medlda = paMedLDAgibbs(config)

  def __check__(me, docs, labels):
    """
    private method,
    the method checks if each word in the docs is in me.words and every label in labels is in me.labels
    """
    warning_word = False
    warning_label = False
    new_docs = list()
    new_labels = list()
    ind = []
    for (di, (doc, label)) in enumerate(zip(docs, labels)):
      new_doc = list()
      for w in doc:
        if not me.words.has_key(w):
          warning_word = True
        else:
          new_doc += [me.words[w]]
      if not me.labels.has_key(label):
        warning_label = True
      elif len(new_doc) > 0:
        new_docs += [new_doc]
        new_labels += [me.labels[label]]
        ind += [di]
    return (new_docs, new_labels, ind)

  def train(me, docs, labels):
    """
    train the model with a mini-batch.
    the method checks if each word in the docs is in me.words and every label in labels is in me.labels
    it removes those words/labels not satisfying the requirement. if so, it emits a warning.

    Parameters
    ______________
    docs: list
      each doc in the list is by itself a list of words.
    labels: list
      a list of corresponding labels.

    """
    (new_docs, new_labels, _) = me.__check__(docs, labels)
    me.medlda.train(new_docs, new_labels)

  def train_with_dataset(me, docs, labels, batchsize):
    """
    train the model on a dataset with given batchsize and one pass.
    the method checks if each word in the docs is in me.words and every label in labels is in me.labels
    it removes those words/labels not satisfying the requirement. if so, it emits a warning.

    the method iteratively chooses a mini-batch from (docs, labels) of batchsize without replace
    until all (docs, labels) have been chosen.

    Parameters
    ______________
    docs: list
      each doc in the list is by itself a list of words.
    labels: list
      a list of corresponding labels.
    batchsize: int
      the mini-batch size
    """
    allind = set(range(len(docs)))
    while len(allind) > 0:
      if len(allind) >= batchsize:
        ind = npr.choice(list(allind), batchsize, replace=False)
      else:
        ind = list(allind)
      print len(allind)
      allind -= set(ind)
      batch_doc = [docs[i] for i in ind]
      batch_label = [labels[i] for i in ind]
      me.train(batch_doc, batch_label)

  def train_with_gml(me, path, batchsize):
    """
    train the model using .gml file, which has the following format:
      line 0:     N         ;  N is the number of documents.
      line 1...N: M L w_1 w_2 w_3 ... w_M ; L is the label of the document, w_1, ..., w_M are words.

    the method iteratively chooses a mini-batch from (docs, labels) of batchsize without replace
    until all (docs, labels) have been chosen.

    Parameters
    _______________
    path:
      path of the .gml file.
    batchsize: int
      the mini-batch size
    """
    (docs, labels, _, _) = read_gml(path)
    me.train_with_dataset(docs, labels, batchsize)

  def infer(me, docs, labels = [], num_sample = 100, point_estimate = False):
    """
    use current MedLDA to classify on new corpus.


    Parameters
    ________________

    docs: list
      each doc in the list is by itself a list of words.
    labels: list, default = []
      a list of corresponding labels.
    num_sample: int, default = 100,
      number of sample used to approximate the posterior.
    point_estimate: bool, default = False,
      whether to use point estimate for test or full Bayesian treatment.


    Return
    ________________
    if labels = [], return (predict_labels, index)
      predict_labels: list
        a list of predicted labels for each document.
      index:
        the indices of docs each prediction corresponds to.

    if labels is not empty, return (predict_labels, index, acc)
      predict_labels: list
        a list of predicted labels for each document.
      index:
        the indices of docs each prediction corresponds to.
      acc:
        test accuracy.
    """
    if labels == []:
      (new_docs, new_labels, ind) = me.__check__(docs, [me.labels.keys()[0]] * len(docs))
    else:
      (new_docs, new_labels, ind) = me.__check__(docs, labels)
    predict_labels = me.medlda.infer(new_docs, new_labels, num_sample, point_estimate)
    if labels == []:
      return (predict_labels, ind)
    else:
      return (predict_labels, ind, me.medlda.testAcc())

  def infer_with_gml(me, path, num_sample, point_estimate=False):
    """
    use current MedLDA to classify on .gml file, which has the following format:

      line 0:     N         ;  N is the number of documents.
      line 1...N: M L w_1 w_2 w_3 ... w_M ; L is the label of the document, w_1, ..., w_M are words.


    Parameters
    ________________

    path:
      path of the .gml file.
    num_sample: int, default = 100,
      number of sample used to approximate the posterior.
    point_estimate: bool, default = False,
      whether to use point estimate for test or full Bayesian treatment.


    Return
    ________________
    if labels = [], return (predict_labels, index)
      predict_labels: list
        a list of predicted labels for each document.
      index:
        the indices of docs each prediction corresponds to.

    if labels is not empty, return (predict_labels, index, acc)
      predict_labels: list
        a list of predicted labels for each document.
      index:
        the indices of docs each prediction corresponds to.
      acc:
        test accuracy.
    """
    (docs, labels, _, _) = read_gml(path)
    return me.infer(docs, labels, num_sample, point_estimate)











