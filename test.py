import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dataset = pd.read_csv('data/Data.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 3].values

# # Taking care of missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transfrom(X[:, 1:3])

# # Encoding categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# lbelencoder_x = LabelEncoder()


from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName('spark-test')
sc = SparkContext(conf=conf)

data = sc.textFile("data/sparktest.data")
ratings = data.map(lambda l:l.split(',')) \
	.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))


rank = 10
numIterations = 10
model = ALS.train(ratings, rank, numIterations)

#evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))

ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))


model.save(sc, "data/output/tmp/myCollaborativeFilter")
# sameModel = MatrixFactorizationModel.load(sc, "data/output/tmp/myCollaborativeFilter")
# 