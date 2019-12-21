# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# from scipy.io import loadmat
# from sklearn.utils import shuffle


# def y2indicator(y):
#     N = len(y)
#     ind = np.zeros((N, 10))
#     for i in range(N):
#         ind[i,y[i]] = 1
#     return ind

# def error_rate(p, t):
#     return np.mean(p != t)

# def flatten(X):
#     N = X.shape[-1]
#     flat = np.zeros((N, 2072))
#     for i in range(N):
#         flat[i] = X[:,:,:,i].reshape(3072)
#     return flat


# docs = ['Well done!',
#         'Good work',
#         'Great effort',
#         'nice work',
#         'Excellent!',
#         'Weak',
#         'Poor effort!',
#         'not good',
#         'poor work',
#         'Could have done better.']
# # define class labels
# labels = array([1,1,1,1,1,0,0,0,0,0])
# vocab_size = 50
# encoded_docs = [one_hot(d, vocab_size) for d in docs]
# print(encoded_docs)

def new_function() -> None:
    print('nice');

if __name__ == '__main__':
    new_function()
