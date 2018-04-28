# import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

h = tf.constant("Hello")
w = tf.constant("world")
hw = h + w

with tf.Session() as sess:
	ans = sess.run(hw)
print(ans)

