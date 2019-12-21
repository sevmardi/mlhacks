import tensorflow as tf
import numpy as np 

# Set random seed for reprouducibility 
np.random.seed(0)
data, labels = tf.contrib.learn.datasets.load_dataset("iris")

num_ele = len(labels)

# use shuffled indexing in shuffle dataset.
shuffled_indc = np.arange(len(labels))

