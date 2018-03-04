# Regularized Neural Network
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from time import time

np.random.random(1327)

df = pd.read_csv('data/mnist.csv')

# training
df_train = df.iloc[:33600, :]
X_train = df_train.iloc[:, 1:].values / 255.
y_train = df_train['label'].values
y_train_onehot = pd.get_dummies(df_train['label']).values


# test
df_test = df.iloc[33600:, :]
X_test = df_test.iloc[:, 1:].values / 255.
y_test = df_test['label'].values

print(y_test)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0, verbose=3)
mode = model.fit(X_train, df_train['label'].values)

# model
y_prediction = model.predict(X_test)
print("\naccuracy", np.sum(y_prediction == df_test[
      'label'].values) / float(len(y_test)))

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

start = time()

model = Sequential()
mode.add(Dense(512, input_shape=(784, )))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

mode.fit(X_train, y_train_onehot)

print('\ntime taken %s seconds' % str(time() - start))

y_prediction = model.predict_classes(X_test)
print "\n\naccuracy", np.sum(y_prediction == y_test) / float(len(y_test))


