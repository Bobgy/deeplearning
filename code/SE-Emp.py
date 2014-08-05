"""
  This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

  Denoising autoencoders are the building blocks for SdA.
  They are based on auto-encodres as the ones used in Bengio et al. 2007.
  An autoencoder takes an input x and first maps it to a hidden representation
  y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
  latent representation y is then mapped back to a "reconstructed" vector
  z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y+b'). The weight
  matrix W' can optionall be constrained such that W' =W^T, in which case
  the autoencoder is said to have tied weights. The network is trained such
  that to minimize the reconstruction error (the error between x and z).

  For the denoising autoencoder, during training, first x is corrupted into
  \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
  of a stochastic mapping. Afterwards y is computed as before (using
  \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
  error is now measured between z and the uncorrupted input x, which is
  computed as the cross-entropy :
    - \sum_{k-1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]

  References :
    - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
      Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
      2008
    - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
      Training of Deep Networks, Advances in Neural Information Processing
      Systems 19, 2007

"""

__docformat__='restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy
import random
import math
from scipy.linalg import pinv

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class LogisticRegression(object):
  """Multi-class Logistic Regression Class

  The logistic regression is fully described by a weight matrix :math:'W'
  and bias vector :math:'b'. Classification is done by projecting data
  points onto a set of hyperplanes, the distance to which is used to
  determine a class membership probability.
  """

  def __init__(self,input,n_in,n_out):
    """ Initialize the parameters of the logistic regression

    :type input: theano.tensor.TensorType
    :param input: symbolic variable that describes the input of the
                  architecture (one minibatch)

    :type n_in: int
    :param n_in: number of input units, the dimension of the sapce in
                 which the datapoints lie
    :type n_out: int
    :param n_out: number of output units, the dimension of the space in
                  which the labels lie

    """

    # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
    self.W=theano.shared(value=numpy.zeros((n_in,n_out),
                                           dtype=theano.config.floatX),
                         name='W',borrow=True)
    # initialize the baises b as a vector of n_out 0s
    self.b=theano.shared(value=numpy.zeros((n_out,),
                                           dtype=theano.config.floatX),
                         name='b',borrow=True)

    # compute vector of class-membership probabilities in symbolic form
    self.p_y_given_x=T.nnet.softmax(T.dot(input,self.W)+self.b)

    # compute prediction as class whose probability is maximal in
    # symbolic form
    self.y_pred=T.argmax(self.p_y_given_x,axis=1)

    # parameters of the model
    self.params=[self.W,self.b]
    """
    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x=T.fmatrix('x')
    y=T.lvector('y')

    # allocate shared variables model params
    b=theano.shared(numpy.zeros((10,)),name='b')
    W=theano.shared(numpy.zeros((784,10)),name='W')

    # symbolic expression for computing the matrix of class-membership probabilities
    # Where:
    # W is a matrix where column-k represent the separation hyper plain for class-k
    # x is a matrix where row-j represents input training sample-j
    # b is a vector where element-k represent the free parameter of hyper plain-k
    p_y_given_x=T.nnet.softmax(T.dot(x,W)+b)

    # compiled Theano function that returns the vector of class-membership
    # probabilities
    get_p_y_given_x=theano.function(inputs=[x],outputs=p_y_given_x)

    # print the probability of some example represented by x_value
    # x_value is not a symbolic variable but a numpy array describing the
    # datapoint
    print 'Probability that x is of class %i is %f' % (i,get_p_y_given_x(x_value)[i])

    # symbolic description of how to compute prediction as class whose probability
    # is maximal
    y_pred=T.argmax(p_y_given_x,axis=1)

    # compiled theano function that returns this value
    classify=theano.function(inputs=[x],outputs=y_pred)
    """

  def negative_log_likelihood(self,y):
    """Return the mean of the negative log-likelihood of the prediction
    of this model under a given target distribution.

    .. math::

      \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
      \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
      \ell (\theta=\{W,b\}, \mathcal{D})

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
              correct label

    Note: we use the mean instead of the sum so that
          the learning rate is less dependent on the batch size
    """
    # y.shape[0] is (symbolically) the number of rows in y, i.e.,
    # number of examples (call it n) in the minibatch
    # T.arange(y.shape[0]) is a symbolic vector which will contain
    # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
    # Log-Probabilities (call it LP) with one row per example and
    # one column per class LP[T.arange(y.shape[0]),y] is a vector
    # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
    # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
    # the mean (across minibatch examples) of the elements in v,
    # i.e., the mean log-likelihood across the minibatch.
    return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
    # note on syntax: T.arange(y.shape[0]) is a vector of integers [0,1,2,...,len(y)].
    # Indexing a matrix M by the two vectors [0,1,...,K], [a,b,...,k] returns the
    # elements M[0,a], M[1,b], ..., M[K,k] as a vector. Here, we use this
    # syntax to retrieve the log-probability of the correct labels, y.

  def errors(self,y):
    """Return a float representing the number of errors in the minibatch
    over the total number of examples of the minibatch ; zero one
    loss over the size of the minibatch

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
              correct label
    """

    # check if y has same dimension of y_pred
    if y.ndim!=self.y_pred.ndim:
      raise TypeError('y should have the same shape as self.y_pred',
                      ('y',target.type,'y_pred',self.y_pred.type))
    # check if y is of the correct datatype
    if y.dtype.startswith('int'):
      # the T.neq operator returns a vector of 0s and 1s, where 1
      # represents a mistake in prediction
      return T.mean(T.neq(self.y_pred,y))
    else:
      raise NotImplementedError()
    
def load_data(dataset):
  """ Loads the dataset

  :type dataset: string
  :param dataset: the path to the dataset (here MNIST)
  """

  #############
  # LOAD DATA #
  #############

  outFile=open("output.txt","a")

  # Download the MNIST dataset if it is not present
  data_dir,data_file=os.path.split(dataset)
  if data_dir=="" and not os.path.isfile(dataset):
    # Check if dataset is in the data directory.
    new_path=os.path.join(os.path.split(__file__)[0],"..","data",dataset)
    if os.path.isfile(new_path) or data_file=='mnist.pkl.gz':
      dataset=new_path

  if (not os.path.isfile(dataset)) and data_file=='mnist.pkl.gz':
    import urllib
    origin='http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    print 'Downloading data from %s' % origin
    iwrite=(("Downloading data from %s\n") % (origin))
    outFile.write(iwrite)
    urllib.urlretrieve(origin,dataset)

  print '... loading data'
  iwrite=("... loading data\n")
  outFile.write(iwrite)

  outFile.close()

  #load the dataset
  f=gzip.open(dataset,'rb')
  train_set,valid_set,test_set=cPickle.load(f)
  f.close()
  #train_set, valid_set, test_set format: tuple(input, target)
  #input is an numpy.ndarray of 2 dimensions (a matrix)
  #witch row's correspond to an example. target is a
  #numpy.ndarray of 1 dimensions (vector)) that have the same length as
  #the number of rows in the input. It should give the target
  #target to the example with the same index in the input.

  def shared_dataset(data_xy,borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x,data_y=data_xy
    shared_x=theano.shared(numpy.asarray(data_x,
                                         dtype=theano.config.floatX),
                           borrow=borrow)
    shared_y=theano.shared(numpy.asarray(data_y,
                                         dtype=theano.config.floatX),
                           borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the label as "floatX" as well
    # ("shared_y" does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # "shared_y" we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x,T.cast(shared_y,'int32')

  test_set_x,test_set_y=shared_dataset(test_set)
  valid_set_x,valid_set_y=shared_dataset(valid_set)
  train_set_x,train_set_y=shared_dataset(train_set)

  rval=[(train_set_x,train_set_y),(valid_set_x,valid_set_y),
        (test_set_x,test_set_y)]
  return rval

class HiddenLayer(object):
  def __init__(self,rng,input,n_in,n_out,U=None,V=None,b=None,
               activation=T.tanh,n_question=10,n_column=10):
    """
    Typical hidden layer of a MLP: units are fully-connected and have
    sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
    and the bias vector b is of shape (n_out,).

    NOTE : The nonlinearity used here is tanh

    Hidden unit activation is given by: tanh(dot(input,W) + b)

    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights

    :type input: theano.tensor.dmatrix
    :param input: a symbolic tensor of shape (n_examples, n_in)

    :type n_in: int
    :param n_in: dimensionality of input

    :type n_out: int
    :param n_out: number of hidden units

    :type activation: theano.Op or function
    :param activation: Non linearity to be applied in the hidden
                       layer
    """
    
    self.input=input

    # 'W' is initialized with 'W_values' which is uniformely sampled
    # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
    # for tanh activation function
    # the output of uniform if converted using asarray to dtype
    # theano.config.floatX so that the code is runable on GPU
    # Note : optimal initialization of weights is dependent on the
    #        activation function used (among other things).
    #        For example, results presented in [Xavier10] suggest that you
    #        should use 4 times larger initial weights for sigmoid
    #        compared to tanh
    #        We have no info for other function, so we use the same as
    #        tanh.
    if V is None:
      V=[]
      for i in xrange(n_column):
        iV_values=numpy.asarray(rng.uniform(
                  low=-numpy.sqrt(6./(n_question+n_out)),
                  high=numpy.sqrt(6./(n_question+n_out)),
                  size=(n_question,n_out)),dtype=theano.config.floatX)
        if activation==theano.tensor.nnet.sigmoid:
          iV_values*=4
        iV=theano.shared(value=iV_values,name='iV',borrow=True)
        V.append(iV)

    if b is None:
      b=[]
      for i in xrange(n_column):
        ib_values=numpy.zeros((n_out,),dtype=theano.config.floatX)
        ib=theano.shared(value=ib_values,name='ib',borrow=True)
        b.append(ib)

    if U is None:
      U=[]
      for i in xrange(n_column):
        iU_values=numpy.zeros((n_in,n_question),dtype=theano.config.floatX)
        iU=theano.shared(value=iU_values,name='iU',borrow=True)
        U.append(iU)

    self.V=V
    self.b=b
    self.U=U

    xoutput=[]
    for i in xrange(n_column):
      ilin_output=T.dot(T.dot(input,self.U[i]),self.V[i])+self.b[i]
      ioutput=(ilin_output if activation is None
               else activation(ilin_output))
      xoutput.append(ioutput)

    output=xoutput[0]
    for i in xrange(n_column-1):
      output=T.add(output,xoutput[i+1])

    self.output=output/n_column

    """
    lin_output=T.dot(input,self.W)+self.b
    self.output=(lin_output if activation is None
                 else activation(lin_output))
    """
    
    # parameters of the model
    self.params=[]
    self.params.extend(self.V)
    self.params.extend(self.b)

class dA(object):
  """Denoising Auto-Encoder class (dA)

  A denoising autoencoders tries to reconstruct the input from a corrupted
  version of it by projecting it first in a latent space and reprojecting
  it afterwards back in the input space. Please refer to Vincent et al.,2008
  for more details. If x is the input then equation (1) computes a partially
  destroyed version of x by means of a stochastic mapping q_D. Equation (2)
  computes the projection of the input into the latent space. Equation (3)
  computes the reconstruction of the input, while equation (4) computes the
  reconstruction error.

  .. math::

    \tilde{x} ~ q_D(\tilde{x}|x)                                (1)

    y = s(W \tilde{x} + b)                                      (2)

    x = s(W' y + b')                                            (3)
    
    L(x,z) = -sum_{k=1}^d [x_k log z_k + (1-x_k) \log( 1-z_k)]  (4)

  """

  def __init__(self,numpy_rng,theano_rng=None,input=None,
               n_visible=784,n_hidden=500,
               U=None,V=None,bhid=None,W=None,bvis=None,
               n_question=78,n_column=10,
               n_heng=28,n_shu=28):
    """
    Initialize the dA class by specifying the number of visible units (the
    dimension d of the input ), the number of hidden units ( the dimension
    d' of the latent or hidden space ) and the corruption level. The
    constructor also receives symbolic variables for the input, weights and
    bias. Such a symbolic variables are useful when, for example the input
    is the result of some computations, or when weights are shared between
    the dA and an MLP layer. When dealing with SdAs this always happens,
    the dA on layer 2 gets as input the output of the dA on layer 1,
    and the weights of the dA are used in the second stage of training
    to construct an MLP.

    :type numpy_rng: numpy.random.RandomState
    :param numpy_rng: number random generator used to generate weights

    :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
    :param theano_rng: Theano random generator; if None is given one is
                       generated based on a seed drawn from 'rng'

    :type input: theano.tensor.TensorType
    :param input: a symbolic description of the input or None for
                  standalone dA

    :type n_visible: int
    :param n_visible: number of visible units

    :type n_hidden: int
    :param n_hidden: number of hidden units

    :type W: theano.tensor.TensorType
    :param W: Theano variable pointing to a set of weights that should be
              shared belong the dA and another architecture; if dA should
              be standalone set this to None

    :type bhid: theano.tensor.TensorType
    :param bhid: Theano variable pointing to a set of biases values (for
                 hidden units) that should be shared belong dA and another
                 architecture; if dA should be standalone set this to None

    :type bvis: theano.tensor.TensorType
    :param bvis: Theano variable pointing to a set of biases values (for
                 visible units) that should be shared belong dA and another
                 architecture; if dA should be standalone set this to None

    """
    self.n_visible=n_visible
    self.n_hidden=n_hidden
    self.n_question=n_question
    self.n_column=n_column
    self.n_heng=n_heng
    self.n_shu=n_shu

    # create a Theano random generator that gives symbolic random values
    if not theano_rng:
      theano_rng=RandomStreams(numpy_rng.randint(2**30))

    # note : W' was written as 'W_prime' and b' as 'b_prime'
    if not W:
      # W is initialized with 'initial_W' which is uniformely sampled
      # from -4*sqrt(6./(n_visible+n_hidden)) and
      # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
      # converted using asarray to dtype
      # theano.config.floatX so that the code is runable on GPU
      initial_W=numpy.asarray(numpy_rng.uniform(
                low=-4*numpy.sqrt(6./(n_hidden+n_visible)),
                high=4*numpy.sqrt(6./(n_hidden+n_visible)),
                size=(n_hidden,n_visible)),dtype=theano.config.floatX)
      W=theano.shared(value=initial_W,name='W',borrow=True)

    if not bvis:
      bvis=theano.shared(value=numpy.zeros(n_visible,
                                           dtype=theano.config.floatX),
                         borrow=True)

    if not bhid:
      bhid=[]
      for i in xrange(n_column):
        ibhid_values=numpy.zeros(n_hidden,dtype=theano.config.floatX)
        ibhid=theano.shared(value=ibhid_values,name='ibhid',borrow=True)
        bhid.append(ibhid)

    self.W=W
    # b corresponds to the bias of the hidden
    self.b=bhid
    # b_prime corresponds to the bias of the visible
    self.b_prime=bvis

    """
    # tied weights, therefore W_prime is W transpose
    self.W_prime=self.W.T
    """

    self.U=U
    self.V=V
    
    self.theano_rng=theano_rng
    # if no input is given, generate a variable representing the input
    if input==None:
      # we use a matrix because we expect a minibatch of several
      # examples, each example being a row
      self.x=T.dmatrix(name='input')
    else:
      self.x=input

    self.params=[]
    self.params.extend(self.U)
    self.params.extend(self.V)
    self.params.extend(self.b)
    self.params.append(self.W)
    self.params.append(self.b_prime)

  def get_corrupted_input(self,input,corruption_level):
    """This function keeps "1-corruption_level" entries of the inputs the
    same and zero-out randomly selected subset of size "coruption_level"
    Note : first argument of theano.rng.binomial is the shape(size) of
           random numbers that it should produce
           second argument is the number of trials
           third argument is the probability of success of any trial

             this will produce an array of 0s and 1s where 1 has a
             probability of 1 - "corruption_level" and 0 with
             "corruption_level"

             The binomial function return int64 data type by
             default. int64 multiplicated by the input
             type(floatX) always return float64. To keep all data
             in floatX when floatX is float32, we set the dtype of
             the binomial to floatX. As in our case the value of
             the binomial is always 0 or 1, this don't change the
             result. This is needed to allow the gpu to work
             correctly as it only support float32 for now.

    """
    return self.theano_rng.binomial(size=input.shape,n=1,
                                    p=1-corruption_level,
                                    dtype=theano.config.floatX)*input
    """
    Help on built-in function binomial:

    Draw samples from a binomial distribution.

    Samples are drawn from a Binomial distribution with specified
    parameters, n trials and p probability of success where
    n an integer >=0 and p is in the interval [0,1]. (n may be
    input as a float, but it is truncated to an integer in use)

    Parameters
    ----------
    n : float (but truncated to an integer)
        parameter, >= 0.
    p : float
        parameter, >=0 and <=1.
    size : {tuple, int}
           Output shape. If the given shape is, e.g., "(m, n, k)", then
           "m * n * k" samples are drawn.

    Returns
    -------
    samples : {ndarray, scalar}
              where the values are all integers in [0, n].

    See Also
    --------
    scipy.stats.distributions.binom : probability density function,
      distribution or cumulative density function, etc.

    Notes
    -----
    The probability density for the Binomial distribution is

    .. math:: P(N) = \binom{n}{N}p^N(1-p)^{n-N},

    where :math:'n' is the number of trials, :math:'p' is the probability
    of success, and :math:'N' is the number of successes.

    When estimating the standard error of a proportion in a population by
    using a random sample, the normal distribution works well unless the
    product p*n <=5, where p = population proportion estimate, and n =
    number of samples, in which case the binomial distribution is used
    instead. For example, a sample of 15 people shows 4 who are left
    handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27*15 = 4,
    so the binomial distribution should be used in this case.

    References
    ----------
    .. [1] Dalgaard, Peter, "Introductory Statistics with R",
           springer-Verlag, 2002.
    .. [2] Glantz, Stanton A. "Primer of Biosatistics.", McGraw-Hill,
           Fifth Edition, 2002.
    .. [3] Lentner, Marvin, "Elementary Applied Statistics", Bogden
           and Quigley, 1972.
    .. [4] Weisstein, Eric W. "Binomial Distribution." From MathWorld--A
           Wofram Web Resource.
           http://mathworld.wolfram.com/BinomialDistribution.html
    .. [5] Wikipedia, "Binomial-distribution",
           http://en.wikipedia.org/wiki/Binomial_distribution

    Examples
    --------
    Draw samples from the distribution:

    >>> n, p = 10, .5 # number of trials, probability of each trial
    >>> s = np.random.binomial(n, p, 1000)
    # result of flipping a coin 10 times, tested 1000 times.

    A real world example. A company drills 9 wild-cat oil exploration
    wells, each with an estimated probability of success of 0.1. All nine
    wells fail. What is the probability of that happening?

    Let's do 20,000 trials of the model, and count the number that
    generate zero positive results.

    >>> sum(np.random.binomial(9,0.1,20000)==0)/20000.
    answer = 0.38885, or 38%.
    """

  def get_hidden_values(self,input):
    """ Computes the values of the hidden layer """
    xoutput=[]
    for i in xrange(self.n_column):
      ilin_output=T.dot(T.dot(input,self.U[i]),self.V[i])+self.b[i]
      ioutput=T.nnet.sigmoid(ilin_output)
      xoutput.append(ioutput)

    output=xoutput[0]
    for i in xrange(self.n_column-1):
      output=T.add(output,xoutput[i+1])
      
    return output/self.n_column

  def get_reconstructed_input(self,hidden):
    """Computes the reconstructed input given the values of the
    hidden layer

    """
    return T.nnet.sigmoid(T.dot(hidden,self.W)+self.b_prime)

  def get_cost_updates(self,corruption_level,learning_rate):
    """ This function computes the cost and the updates for one trainng
    step of the dA """

    tilde_x=self.get_corrupted_input(self.x,corruption_level)
    y=self.get_hidden_values(tilde_x)
    z=self.get_reconstructed_input(y)
    # note : we sum over the size of a datapoint; if we are using
    #        minibatches, L will be a vector, with one entry per
    #        example in minibatch
    L=-T.sum(self.x*T.log(z)+(1-self.x)*T.log(1-z),axis=1)
    # note : L is now a vector, where each element is the
    #        cross-entropy cost of the reconstruction of the
    #        corresponding example of the minibatch. We need to
    #        compute the average of all these to get the cost of
    #        the minibatch
    cost=T.mean(L)

    # compute the gradients of the cost of the 'dA' with respect
    # to its parameters
    gparams=T.grad(cost,self.params)
    # generate the list of updates
    updates=[]
    for param,gparam in zip(self.params,gparams):
      updates.append((param,param-learning_rate*gparam))

    return (cost,updates)

  def expo(self,a,b):
    
    ix=a/self.n_shu
    iy=a%self.n_shu
    if iy==0:
      iy=self.n_shu
    else:
      ix=ix+1
  
    jx=b/self.n_shu
    jy=b%self.n_shu
    if jy==0:
      jy=self.n_shu
    else:
      jx=jx+1

    s=math.pow((ix-jx),2)+math.pow((iy-jy),2)
    s=(-1.0)*s/(2*(math.pow(1.0,2)))
    s=math.exp(s)

    return s

  def re_u1(self):

    x=numpy.zeros((self.n_visible,self.n_visible),dtype=theano.config.floatX)
    for j in xrange(self.n_visible):
      for k in xrange(self.n_visible):
        x[j][k]=self.expo(j+1,k+1)
    
    for i in xrange(self.n_column):
      
      k_lit=numpy.zeros((self.n_question,self.n_visible),dtype=theano.config.floatX)
      select_list=range(1,self.n_visible+1)
      alpha=random.sample(select_list,self.n_question)
      alpha=sorted(alpha)
      for j in xrange(self.n_question):
        k_lit[j]=x[alpha[j]-1]
          
      """
      print select_list
      print alpha
      print len(alpha)
      print k_lit
      print k_lit.shape
      return
      """
      
      k_lar=numpy.zeros((self.n_question,self.n_question),dtype=theano.config.floatX)
      for j in xrange(self.n_question):
        for k in xrange(self.n_question):
          k_lar[j][k]=k_lit[j][(alpha[k]-1)]

      """
      print k_lar
      print k_lar.shape
      """

      i_lar=numpy.eye(self.n_question,self.n_question,0,dtype=theano.config.floatX)

      """
      print i_lar
      print i_lar.shape
      """

      s=numpy.add(k_lar,0.01*i_lar)
      s=pinv(s)
      s=numpy.dot(k_lit.T,s)

      """
      print s
      print s.shape
      """
      
      self.U[i].set_value(s)
      
  def re_u2(self,all_out):
    
    x=numpy.cov(all_out[0:5000].T)

    for i in xrange(self.n_column):
      
      k_lit=numpy.zeros((self.n_question,self.n_visible),dtype=theano.config.floatX)
      select_list=range(1,self.n_visible+1)
      alpha=random.sample(select_list,self.n_question)
      alpha=sorted(alpha)
      for j in xrange(self.n_question):
        k_lit[j]=x[alpha[j]-1]

      k_lar=numpy.zeros((self.n_question,self.n_question),dtype=theano.config.floatX)
      for j in xrange(self.n_question):
        for k in xrange(self.n_question):
          k_lar[j][k]=k_lit[j][(alpha[k]-1)]

      i_lar=numpy.eye(self.n_question,self.n_question,0,dtype=theano.config.floatX)

      s=numpy.add(k_lar,0.01*i_lar)
      s=pinv(s)
      s=numpy.dot(k_lit.T,s)

      self.U[i].set_value(s)

class SdA(object):
  """Stacked denoising auto-encoder class (SdA)

  A stacked denoising autoencoder model is obtained by stacking several
  dAs. The hidden layer of the dA at layer 'i' becomes the input of
  the dA at layer 'i+1'. The first layer dA gets as input the input of
  the SdA, and the hidden layer of the last dA represents the output.
  Note that after pretraining, the SdA is dealt with as a normal MLP,
  the dAs are only used to initialize the weights.
  """

  def __init__(self,numpy_rng,theano_rng=None,n_ins=784,
               hidden_layers_sizes=[500,500],n_outs=10,
               corruption_levels=[0.1,0.1],n_column=10,
               n_heng=28,n_shu=28,irate=1.0):
    """ This class is made to support a variable number layers.

    :type numpy_rng: numpy.random.RandomState
    :param numpy_rng: numpy random number generator used to draw initial
                      weights

    :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
    :param theano_rng: Theano random generator; if None is given one is
                       generated based on as seed drawn from 'rng'

    :type n_ins: int
    :param n_ins: dimensions of the input to the sdA

    :type n_layers_sizes: list of ints
    :param n_layers_sizes: intermediate layers size, must contain
                           at least one value

    :type n_outs: int
    :param n_outs: dimension of the output of the network

    :type corruption_levels: list of float
    :param corruption_levels: amount of corruption to use for each
                              layer
    """

    self.sigmoid_layers=[]
    self.dA_layers=[]
    self.params=[]
    self.n_layers=len(hidden_layers_sizes)

    assert self.n_layers>0

    if not theano_rng:
      theano_rng=RandomStreams(numpy_rng.randint(2**30))
    # allocate symbolic variables for the data
    self.x=T.matrix('x') # the data is presented as rasterized images
    self.y=T.ivector('y') # the labels are presented as 1D vector of
                          # [int] labels

    # The SdA is an MLP, for which all weights of intermediate layers
    # are shared with a different denoising autoencoders
    # We will first construct the SdA as a deep multilayer perceptron,
    # and when constructing each sigmoidal layer we also construct a
    # denoising autoencoder that shares weights with that layer
    # During pretraining we will train these autoencoders (which will
    # lead to chainging the weights of the MLP as well)
    # During finetuning we will finish training the SdA by doing
    # stochastich gradient descent on the MLP

    for i in xrange(self.n_layers):
      # construct the sigmoidal layer

      # the size of the input is either the number of hidden units of
      # the layer below or the input size if we are on the first layer
      if i==0:
        input_size=n_ins
      else:
        input_size=hidden_layers_sizes[i-1]

      # the input to this layer is either the activation of the hidden
      # layer below or the input of the Sda if you are on the first
      # layer
      if i==0:
        layer_input=self.x
      else:
        layer_input=self.sigmoid_layers[-1].output

      n_question=int(irate*input_size/n_column)

      sigmoid_layer=HiddenLayer(rng=numpy_rng,
                                input=layer_input,
                                n_in=input_size,
                                n_out=hidden_layers_sizes[i],
                                activation=T.nnet.sigmoid,
                                n_question=n_question,
                                n_column=n_column)     
      # add the layer to our list of layers
      self.sigmoid_layers.append(sigmoid_layer)
      # its arguably a philosophical question...
      # but we are going to only declare that the parameters of the
      # sigmoid_layers are parameters of the StackedDAA
      # the visible biases in the dA are parameters of those
      # dA, but not the SdA
      self.params.extend(sigmoid_layer.params)

      # Construct a denoising autoencoder that shared weights with this
      # layer
      dA_layer=dA(numpy_rng=numpy_rng,
                  theano_rng=theano_rng,
                  input=layer_input,
                  n_visible=input_size,
                  n_hidden=hidden_layers_sizes[i],
                  U=sigmoid_layer.U,
                  V=sigmoid_layer.V,
                  bhid=sigmoid_layer.b,
                  n_question=n_question,
                  n_column=n_column,
                  n_heng=n_heng,n_shu=n_shu)
      
      if i==0:
        dA_layer.re_u1()
        
      self.dA_layers.append(dA_layer)

    # We now need to add a logistic layer on top of the MLP
    self.logLayer=LogisticRegression(
                  input=self.sigmoid_layers[-1].output,
                  n_in=hidden_layers_sizes[-1],n_out=n_outs)

    self.params.extend(self.logLayer.params)
    # construct a function that implements one step of finetunining

    # compute the cost for second phase of training,
    # defined as the negative log likelihood
    self.finetune_cost=self.logLayer.negative_log_likelihood(self.y)
    # compute the gradients with respect to the model parameters
    # symbolic variable that points to the number of errors made on the
    # minibatch given by self.x and self.y
    self.errors=self.logLayer.errors(self.y)

  def pretraining_functions(self,train_set_x,batch_size):
    ''' Generates a list of functions, each of them implementing one
    setp in trainnig the dA corresponding to the layer with same index.
    The function will require as input the minibatch index, and to train
    a dA you just need to iterate, calling the corresponding function on
    all minibatch indexes.

    :type train_set_x: theano.tensor.TensorType
    :param train_set_x: Shared variable that contains all datapoints used
                        for training the dA

    :type batch_size: int
    :param batch_size: size of a [mini]batch

    :type learning_rate: float
    :param learning_rate: learning rate used during training for any of
                          the dA layers
    '''

    # index to a [mini]batch
    index=T.lscalar('index') # index to a minibatch
    corruption_level=T.scalar('corruption') # % of corruption to use
    learning_rate=T.scalar('lr') # learning rate to use
    # number of batches
    n_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
    # begining of a batch, given 'index'
    batch_begin=index*batch_size
    # ending of a batch given 'index'
    batch_end=batch_begin+batch_size

    pretrain_fns=[]
    for dA in self.dA_layers:
      # get the cost and the updates list
      cost,updates=dA.get_cost_updates(corruption_level,
                                       learning_rate)
      # compile the theano function
      fn=theano.function(inputs=[index,
                                 theano.Param(corruption_level,default=0.2),
                                 theano.Param(learning_rate,default=0.1)],
                         outputs=cost,
                         updates=updates,
                         givens={self.x:train_set_x[batch_begin:batch_end]})
      # append 'fn' to the list of functions
      pretrain_fns.append(fn)

    return pretrain_fns

  def build_fix_functions(self,train_set_x,batch_size):
    
    index=T.lscalar('index')
    n_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
    batch_begin=index*batch_size
    batch_end=batch_begin+batch_size

    fix_u_fns=[]
    for sigmoid in self.sigmoid_layers:
      all_out=sigmoid.output
      fn=theano.function(inputs=[index],
                         outputs=all_out,
                         givens={self.x:train_set_x[batch_begin:batch_end]})
      fix_u_fns.append(fn)

    return fix_u_fns

  def build_finetune_functions(self,datasets,batch_size,learning_rate):
    '''Generates a function 'train' that implements one step of
    finetuning, a function 'validate' that computes the error on
    a batch from the validation set, and a function 'test' that
    computes the error on a batch from the testing set

    :type datasets: list of pairs of theano.tensor.TensorType
    :param datasets: It is a list that contain all the datasets;
                     the has to contain three pairs, 'train',
                     'valid', 'test' in this order, where each pair
                     is formed of two Theano variables, one for the
                     datapoints, the other for the labels

    :type batch_size: int
    :param batch_size: size of a minibatch

    :type learning_rate: float
    :param learning_rate: learning rate used during finetune stage
    '''

    (train_set_x,train_set_y)=datasets[0]
    (valid_set_x,valid_set_y)=datasets[1]
    (test_set_x,test_set_y)=datasets[2]

    # compute number of minibatches for training, validation and testing
    n_valid_batches=valid_set_x.get_value(borrow=True).shape[0]
    n_valid_batches/=batch_size
    n_test_batches=test_set_x.get_value(borrow=True).shape[0]
    n_test_batches/=batch_size

    index=T.lscalar('index') # index to a [mini]batch

    # compute the gradients with respect to the model parameters
    gparams=T.grad(self.finetune_cost,self.params)

    # compute list of fine-tuning updates
    updates=[]
    for param,gparam in zip(self.params,gparams):
      updates.append((param,param-gparam*learning_rate))

    train_fn=theano.function(inputs=[index],
                             outputs=self.finetune_cost,
                             updates=updates,
                             givens={
                               self.x:train_set_x[index*batch_size:
                                                  (index+1)*batch_size],
                               self.y:train_set_y[index*batch_size:
                                                  (index+1)*batch_size]},
                             name='train')

    test_score_i=theano.function([index],self.errors,
                                 givens={
                                   self.x:test_set_x[index*batch_size:
                                                     (index+1)*batch_size],
                                   self.y:test_set_y[index*batch_size:
                                                     (index+1)*batch_size]},
                                 name='test')

    valid_score_i=theano.function([index],self.errors,
                                  givens={
                                    self.x:valid_set_x[index*batch_size:
                                                       (index+1)*batch_size],
                                    self.y:valid_set_y[index*batch_size:
                                                       (index+1)*batch_size]},
                                  name='valid')

    # Create a function that scans the entire validation set
    def valid_score():
      return [valid_score_i(i) for i in xrange(n_valid_batches)]

    # Create a function that scans the entire test set
    def test_score():
      return [test_score_i(i) for i in xrange(n_test_batches)]

    return train_fn,valid_score,test_score

def test_SdA(finetune_lr=0.1,pretraining_epochs=15,
             pretrain_lr=0.001,training_epochs=1000,
             dataset='mnist.pkl.gz',batch_size=1):
  """
  Demonstrates how to train and test a stochastic denoising autoencoder.

  This is demonstrated on MNIST.

  :type learning_rate: float
  :param learning_rate: learning rate used in the finetune stage
                        (factor for the stochastic gradient)

  :type pretraining_epochs: int
  :param pretraining_epochs: number of epoch to do pretraining

  :type pretrain_lr: float
  :param pretrain_lr: learning rate to be used during pre-training

  :type n_iter: int
  :param n_iter: maximal number of iterations ot run the optimizer

  :type dataset: string
  :param dataset: path the the pickled dataset

  """

  outFile=open("output.txt","a")

  datasets=load_data(dataset)

  train_set_x,train_set_y=datasets[0]
  valid_set_x,valid_set_y=datasets[1]
  test_set_x,test_set_y=datasets[2]

  # compute number of minibatches for training, validation and testing
  n_train_batches=train_set_x.get_value(borrow=True).shape[0]
  n_train_batches/=batch_size

  # numpy random generator
  numpy_rng=numpy.random.RandomState(89677)
  print '... building the model'
  iwrite=("... building the model\n")
  outFile.write(iwrite)
  
  # construct the stacked denoising autoencoder class
  sda=SdA(numpy_rng=numpy_rng,n_ins=28*28,
          hidden_layers_sizes=[500,500],
          n_outs=10,n_column=10,
          n_heng=28,n_shu=28,
          irate=0.15)

  #########################
  # PRETRAINING THE MODEL #
  #########################
  print '... getting the pretraining functions'
  iwrite=("... getting the pretraining functions\n")
  outFile.write(iwrite)
  pretraining_fns=sda.pretraining_functions(train_set_x=train_set_x,
                                            batch_size=batch_size)

  print '... pre-training the model'
  iwrite=("... pre-training the model\n")
  outFile.write(iwrite)

  fixing_u_fns=sda.build_fix_functions(train_set_x=train_set_x,
                                      batch_size=batch_size)
  
  start_time=time.clock()
  s_time=time.time()
  ## Pre-train layer-wise
  corruption_levels=[.1,.2]
  for i in xrange(sda.n_layers):
    if i>=1:
      for batch_index in xrange(n_train_batches):
        iall_out=fixing_u_fns[i-1](index=batch_index)
        if batch_index==0:
          all_out=numpy.zeros(((n_train_batches*batch_size),iall_out.shape[1]),
                              dtype=theano.config.floatX)
        all_out[(batch_index*batch_size):((batch_index+1)*batch_size)]=iall_out
      """
      print "hello"
      print all_out
      print all_out.shape
      print "hello"
      """
      sda.dA_layers[i].re_u2(all_out)
      
    # go through pretraining epochs
    for epoch in xrange(pretraining_epochs):
      # go through the training set
      c=[]
      for batch_index in xrange(n_train_batches):
        c.append(pretraining_fns[i](index=batch_index,
                                    corruption=corruption_levels[i],
                                    lr=pretrain_lr))
      print 'Pre-training layer %i, epoch %d, cost ' % (i,epoch),
      iwrite=(("Pre-training layer %i, epoch %d, cost ") % (i,epoch))
      outFile.write(iwrite)
      print numpy.mean(c)
      iwrite=(str(numpy.mean(c))+"\n")
      outFile.write(iwrite)

  end_time=time.clock()
  e_time=time.time()

  print >> sys.stderr,('The pretraining code for file '+
                       os.path.split(__file__)[1]+
                       ' ran for %.2fm' % ((end_time-start_time)/60.))
  iwrite=(("The pretraining code for file "+
           os.path.split(__file__)[1]+
           " ran for %.2fm\n") % ((end_time-start_time)/60.))
  outFile.write(iwrite)

  print >> sys.stderr,('The pretraining code for file '+
                       os.path.split(__file__)[1]+
                       ' ran for %.2fm' % ((e_time-s_time)/60.))
  iwrite=(("The pretraining code for file "+
           os.path.split(__file__)[1]+
           " ran for %.2fm\n") % ((e_time-s_time)/60.))
  outFile.write(iwrite)

  ########################
  # FINETUNING THE MODEL #
  ########################

  # get the training, validation and testing function for the model
  print '... getting the finetuning functions'
  iwrite=("... getting the finetuning functions\n")
  outFile.write(iwrite)
  train_fn,validate_model,test_model=sda.build_finetune_functions(
                                     datasets=datasets,batch_size=batch_size,
                                     learning_rate=finetune_lr)

  print '... finetunning the model'
  iwrite=("... finetunning the model\n")
  outFile.write(iwrite)
  # early-stopping parameters

  patience=10*n_train_batches # look as this many examples regardless
  patience_increase=2. # wait this much longer when a new best is
                       # found
  improvement_threshold=0.995 # a relative improvement of this much is
                              # considered significant

  validation_frequency=min(n_train_batches,patience/2)
                       # go through this many
                       # minibatche before checking the network
                       # on the validation set; in this case we
                       # check every epoch

  best_params=None
  best_validation_loss=numpy.inf
  test_score=0.
  start_time=time.clock()
  s_time=time.time()

  done_looping=False
  epoch=0

  while (epoch<training_epochs) and (not done_looping):
    epoch=epoch+1
    c=[]
    for minibatch_index in xrange(n_train_batches):
      minibatch_avg_cost=train_fn(minibatch_index)
      iter=(epoch-1)*n_train_batches+minibatch_index+1

      c.append(minibatch_avg_cost)

      """
      print "epoch:",epoch
      iwrite=("epoch: "+str(epoch)+"\n")
      outFile.write(iwrite)
      print "iter:",iter+1
      iwrite=("iter: "+str(iter+1)+"\n")
      outFile.write(iwrite)
      print "minibatch_avg_cost:",minibatch_avg_cost
      iwrite=("minibatch_avg_cost: "+str(minibatch_avg_cost)+"\n")
      outFile.write(iwrite)
      """

      if (iter%validation_frequency)==0:
        validation_losses=validate_model()
        this_validation_loss=numpy.mean(validation_losses)
        print ('epoch %i, minibatch %i/%i, validation error %f %%' %
               (epoch,minibatch_index+1,n_train_batches,
                this_validation_loss*100.))
        iwrite=(("epoch %i, minibatch %i/%i, validation error %f %%\n") %
                (epoch,minibatch_index+1,n_train_batches,
                 this_validation_loss*100.))
        outFile.write(iwrite)

        print 'epoch %d, cost ' % (epoch),
        iwrite=(("epoch %d, cost ") % (epoch))
        outFile.write(iwrite)
        print numpy.mean(c)
        iwrite=(str(numpy.mean(c))+"\n")
        outFile.write(iwrite)

        # if we got the best validation score until now
        if this_validation_loss<best_validation_loss:

          #improve patience if loss improvement is good enough
          if (this_validation_loss<best_validation_loss*
              improvement_threshold):
            print (("this_validation_loss: %f %%") % (this_validation_loss*100.))
            iwrite=(("this_validation_loss: %f %%\n") % (this_validation_loss*100.))
            outFile.write(iwrite)
            print (("best_validation_loss: %f %%") % (best_validation_loss*100.))
            iwrite=(("best_validation_loss: %f %%\n") % (best_validation_loss*100.))
            outFile.write(iwrite)
            print "patience:",patience
            iwrite=("patience: "+str(patience)+"\n")
            outFile.write(iwrite)
            print "iter:",iter
            iwrite=("iter: "+str(iter)+"\n")
            outFile.write(iwrite)
            
            patience=max(patience,(iter*patience_increase))
            
            print "patience:",patience
            iwrite=("patience: "+str(patience)+"\n")
            outFile.write(iwrite)

          # save best validation score and iteration number
          best_validation_loss=this_validation_loss
          best_iter=iter

          # test it on the test set
          test_losses=test_model()
          test_score=numpy.mean(test_losses)
          print (('    epoch %i, minibatch %i/%i, test error of '
                  'best model %f %%') %
                 (epoch,minibatch_index+1,n_train_batches,
                  test_score*100.))
          iwrite=(("    epoch %i, minibatch %i/%i, test error of "
                   "best model %f %%\n") %
                  (epoch,minibatch_index+1,n_train_batches,
                   test_score*100.))
          outFile.write(iwrite)

      if patience<=iter:
        done_looping=True
        break

  end_time=time.clock()
  e_time=time.time()
  print (('Optimization complete. Best validation score of %f %% '
          'obtained at iteration %i, with test performance %f %%') %
         (best_validation_loss*100.,best_iter,test_score*100.))
  iwrite=(("Optimization complete. Best validation score of %f %% "
           "obtained at iteration %i, with test performance %f %%\n") %
          (best_validation_loss*100.,best_iter,test_score*100.))
  outFile.write(iwrite)
  print 'The code run for %d epochs, with %f epochs/min' % (
        epoch,1.*epoch/((end_time-start_time)/60.))
  iwrite=(("The code run for %d epochs, with %f epochs/min\n") % (
          epoch,1.*epoch/((end_time-start_time)/60.)))
  outFile.write(iwrite)
  print >> sys.stderr,('The training code for file '+
                       os.path.split(__file__)[1]+
                       ' ran for %.2fm' % ((end_time-start_time)/60.))
  iwrite=(("The training code for file "+
           os.path.split(__file__)[1]+
           " ran for %.2fm\n") % ((end_time-start_time)/60.))
  outFile.write(iwrite)

  print "patience:",patience
  iwrite=("patience: "+str(patience)+"\n")
  outFile.write(iwrite)
  print "iter:",iter
  iwrite=("iter: "+str(iter)+"\n")
  outFile.write(iwrite)
  print "done_looping:",done_looping
  iwrite=("done_looping: "+str(done_looping)+"\n")
  outFile.write(iwrite)

  print (('Optimization complete. Best validation score of %f %% '
          'obtained at iteration %i, with test performance %f %%') %
         (best_validation_loss*100.,best_iter,test_score*100.))
  iwrite=(("Optimization complete. Best validation score of %f %% "
           "obtained at iteration %i, with test performance %f %%\n") %
          (best_validation_loss*100.,best_iter,test_score*100.))
  outFile.write(iwrite)
  print 'The code run for %d epochs, with %f epochs/min' % (
        epoch,1.*epoch/((e_time-s_time)/60.))
  iwrite=(("The code run for %d epochs, with %f epochs/min\n") % (
          epoch,1.*epoch/((e_time-s_time)/60.)))
  outFile.write(iwrite)
  print >> sys.stderr,('The training code for file '+
                       os.path.split(__file__)[1]+
                       ' ran for %.2fm' % ((e_time-s_time)/60.))
  iwrite=(("The training code for file "+
           os.path.split(__file__)[1]+
           " ran for %.2fm\n") % ((e_time-s_time)/60.))
  outFile.write(iwrite)

  outFile.close()

if __name__=='__main__':
  test_SdA()

"""
Running the Code
The user can run the code by calling:
  python code/SdA.py
By default the code runs 15 pre-training epochs for each layer, with
a batch size of 1. The corruption level for the first layer is 0.1,
for the second 0.2 and 0.3 for the third. The pretraining learning
rate is was 0.001 and the finetuning learning rate is 0.1.
Pre-training takes 585.01 minutes, with an average of 13 minutes per
epoch. Fine-tuning is completed after 36 epochs in 444.2 minutes,
with an average of 12.34 minutes per epoch. The final validation
score is 1.39% with a testing score of 1.3%. These results were
obtained on a machine with an Intel Xeon E5430 @ 2.66GHz CPU, with a
single-threaded GotoBLAS.
"""

"""
Tips and Tricks
One way to improve the running time of your code (given that you
have sufficient memory available), is to compute how the network, up
to layer k-1, transforms your data. Namely, you start by training
your first layer dA. Once it is trained, you can compute the hidden
units values for every datapoint in your dataset and store this as a
new dataset that you will use to train the dA corresponding to
layer 2. Once you trained the dA for layer 2, you compute, in a
similar fashion, the dataset for layer 3 and so on. You can see now,
that at this point, the dAs are trained individually, and the just
provide (one to the other) a non-linear transformation of the input.
Once all dAs are trained, you can start fine-tunning the model.
"""
