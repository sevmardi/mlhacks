# A model to describe people spend more time on site (e.g social media)
# Alpha is and beta are params we could tweak
# x_i is the number of friends a user has.
# y_i is the numnber of minutes user(s) spends on the site.
# epislon is the error term

import numpy as np
import random

def predict(alpha, beta, x_i):
    return beta * xi + alpha


def error(alpha, beta, x_i, y_i):
    """The error from predicting beta * x_i + alpha when the actual value is y_i"""
    return y_i - predict(alpha, beta, x_i)


def sum_of_sequared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2 for x_i, y_i in zip(x, y))


def de_mean(x):
    x_bar = np.mean(x)
    return [x_i - x_bar for x_i in x]


def least_sequare_fit(x, y):
    """the	total	squared	variation	of	y_i's	from	their	mean"""
    return sum(v ** 2 for v in de_mean(y))


def total_sum_of_sequares(y):
    """the fraction of variation in y captured by the model , which equals 1 - the fraction of variation in y not captured in y not captured by the model"""
    return 1.0 - (sum_of_sequared_errors(alpha, beta, x, y) / total_sum_of_sequares(y))


def sequared_error(x_i, y_i, theta):
	alpha, beta = theta
	return error(alpha, beta, x_i, y_i) ** 2


def sequared_error_gradient(x_i, y_i, theta):
	alpha, beta = theta
	return [-2 * error(alpha, beta, x_i, y_i), -2 * error(alpha, beta, x_i,y_i) * x_i]

random.seed(0)
theta = [random.random(), random.random()]
alpha, beta =  minimize_stochastic(sequared_error, sequared_error_gradient, num_frineds_good, daily_minutes_good, theta, 0.0001)


print(alpha)
print(beta)