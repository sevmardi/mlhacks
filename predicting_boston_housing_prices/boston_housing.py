import numpy as np
import pandas as pd
import matplotlib.pyplot as pls
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import ShuffleSplit

data = pd.read_csv('housing.csv')

housing_prices = data['MDEV']
housing_features = data.drop('MDEV', axis=1)


# print("Boston Housing dataset loaded successfully")

# Number of houses in the dataset
total_houses = housing_prices.count

# Number of features in the dataset
total_features = housing_features.shape

# Minimum housing value in the dataset
minimum_price = housing_prices.min()

# Maximum housing value in the dataset
maximum_price = housing_prices.max()

# Mean house value of the dataset
mean_price = housing_prices.mean()

# Median house value of the dataset
# median_price = housing_prices.median()
median_price = np.median(housing_prices)


# Standard deviation of housing values of the dataset
std_dev = housing_prices.std()

# print("Boston Housing dataset statistics (in $1000's):\n")
# print("Total number of houses:", total_houses)
# print("Total number of features:", total_features)
# print("Minimum house price:", minimum_price)
# print("Maximum house price:", maximum_price)
# print("Mean house price: {0:.3f}".format(mean_price))
# print("Median house price:", median_price)
# print("Standard deviation of house price: {0:.3f}".format(std_dev))


def shuffle_split_data(X, y):
    X_train, y_train, X_test, y_test = ShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    # X_train = None
    # y_train = None
    # X_test = None
    # y_test = None
    return X_train, y_train, X_test, y_test

shuffle_split_data(housing_features, housing_prices)
# try:
# 	X_train, y_train, X_test, y_test = shuffle_split_data(housing_features, housing_prices)
# 	print("Successfully shuffled and split the data!")

# except:
# 	print("Something went wrong with shuffling and splitting the data.")


def performance_metric(y_true, y_predict):
	"""Calculates and returns the total error between true and predicted values
	based on the performance metric chosen by the student

	Arguments:
		y_true  - Perform a total error calculation between the true values of the y
		y_predict - predicted values
	"""

	score = r2(y_true, y_predict)

	return score
