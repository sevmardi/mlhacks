import numpy as np
import pandas as pd
# from itertools import izip
from sklearn.preprocessing import StandardScaler
from time import time

np.random.seed(1337)

df = pd.read_csv('data/mnist.csv')

# for item in df.iloc[0, 1:].values.reshape(28,28)  / 26:
# 	 print ('').join(str(list(item)).split(' '))

import matplotlib.pyplot as plt

# plt.imshow(df.iloc[0, 1:].values.reshape(28,28), cmap=plt.get_cmap('gray', 5))
# plt.show()


df_train = df.iloc[:33600, :]

X_train = df_train.iloc[:, 1:].values / 255.
y_train = df_train['label'].values
y_train_onehot = pd.get_dummies(df_train['label']).values



df_test = df.iloc[33600:, :]
X_test = df_test.iloc[:, 1:].values / 255.
y_test = df_test['label'].values



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=0, verbose=3)
model = model.fit(X_train, df_train['label'].values)

y_prediction = model.predict(X_test)
print("\naccuray", np.sum(y_prediction == df_test['label'].values) / float(len(y_test)))

from keras.models import Sequential
from keras.layers import Dense, Activation

start = time()

model = Sequential()
mode.add(Dense(input_dim=784, output_dim=10))
mode.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
mode.fit(X_train, y_train_onehot)
print('\ntime taken % seconds' % str(time() - start))
y_prediction = model.predict_classes(X_test)
print "\n\naccuracy", np.sum(y_prediction == y_test) / float(len(y_test))


#2 layer NN 
model.add(Dense(input_dim=784, output_dim=100))
mode.add(Dense(output_dim=10))
mode.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train_onehot)

print('\ntime taken %s seconds' % str(time) - start())
y_prediction = model.predict_classes(X_test)
print "\n\naccuracy", np.sum(y_prediction == y_test) / float(len(y_test))


#3layer NN 
model.add(Dense(input_dim=784, output_dim=100))
model.add(Dense(output_dim=100))
model.add(Dense(output_dim=10))

model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train_onehot)



print('\ntime taken %s seconds' % str(time) - start())
y_prediction = model.predict_classes(X_test)
print "\n\naccuracy", np.sum(y_prediction == y_test) / float(len(y_test))





