import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



x = 2
y = 3
add_op = tf.add(x,y)
mul_op = tf.multiply(x,y)

useless = tf.pow(x, mul_op)
pow_op = tf.pow(add_op, mul_op)

with tf.Session() as sess:
	z = sess.run(pow_op)



def scale(data_matrix):
    """returns the mean and standard deviations of each columns"""
    num_rows, num_cols = shape(data_matrix)
    means = [mean]


def split_data(data, prob):
    result = [], []
    for row in data:
        result[0 if random.random() < prob else 1].append(row)

    return result


def train_test_split(x, y, test_pct):
    data = zip(x, y)
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test


