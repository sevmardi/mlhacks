#Convolutional Neural Network
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from time import time

np.random.seed(1337)

df = pd.read_csv('data/mnist.csv')


#train
df_train = df.iloc[:33600, :]
X_train = df_train.iloc[:, 1:].values / 255.
y_train = df_train['label'].values
y_train_onehot = pd.get_dummies(df_train['label']).values

#test
df_test = df.iloc[33600:, :]
X_test = df_test.iloc[:, 1:].values / 255.
y_test = df_test['label'].values

#model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0, verbose=3)
model = model.fit(X_train, df_train['label'].values)
y_prediction = model.predict(X_train)

print('\naccuracy', np.sum(y_prediction == df_test['label'].values) / float(len(y_test)))


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten

start = time()

img_rows, img_cols = 28, 28
nb_filters = 32
pool_size = (2,2)
kernel_size = (3,3)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

model = Sequential()
mode.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))

model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X_train, y_train_onehot)

print('\ntime taken %s seconds' % str(time() - start))

y_prediction = model.predict_classes(X_test)

print('\naccuracy', np.sum(y_prediction == y_test) / float(len(y_test)))



