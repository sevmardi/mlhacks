import numpy as np
import theano 
import theano.tensor as T
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle 
from util import relu, error_rate, getKaggleMNIST, init_weights 

class AutoEncoder(object):
    def __init__(self, M, an_id):
        self.M = M
        self.id = an_id

    def fit(self, X, learning_rate=0.5, mu=0.99, epochs=1, batch_sz=100 show_fig=False):
        N,D = X.shape
        n_batch = N  / batch_sz

        W0 = init_weights((D, self.M))

        self.W = theano.shared(W0, 'W_%s' % self.id)

        self.bh = theano.shared(np.zeros(self.M), 'tb_%s' % self.id)
        self.bo = theano.shared(np.zeros(D), 'bo_%s'  % self.id)

        self.params = [self.W, self.bh, self.bo]

        self.forward_params = [self.W, self.bh]

        self.dW = theano.shared(np.zeros(W0.shape), 'dW_%s' % self.id)
        self.dbh = theano.shared(np.zeros(self.M), 'dbh_%s' % self.id)
        self.dbo = theano.shared(np.zeros(D), 'dbo_%s' % self.id)

        self.dparams = [self.dW, self.dbh, self.dbo]
        self.forward_params = [self.dW, self.dbh]


        X_in = T.matrix('')
