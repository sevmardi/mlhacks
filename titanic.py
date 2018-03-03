import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.ensemble import RandomForestClassifier


np.random.seed(1337)

df = pd.read_csv('datasets/titanic.csv')
# print(df.head())

df_train = df.iloc[:712, :]

scaler = StandardScaler()
features = ['Pclass', 'Sex', 'Age', 'Fare']

# Train
x_train = scaler.fit_transform(df_train[features].values)
y_train = df_train['Survived'].values
y_train_onehot = pd.get_dummies(df_train['Survived']).values

# Test
df_test = df.iloc[712:, :]
x_test = scaler.transform(df_test[features].values)
y_test = df_test['Survived'].values


# model
model = RandomForestClassifier(random_state=0, verbose=3)
model = model.fit(x_train, y_train)
y_predict = model.predict(x_test)

print("\naccuracy", np.sum(y_prediction == y_test) / float(len(y_test)))


# 1-layer Neural Network
###############################
from keras.models import Sequential
from keras.layers import Dense, Activation


start = time()

model = Sequential()
model.add(Dense(input_dim=4, output_dim=2))
model.add(Activation("softmax"))


model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train_onehot)

print('\ntime taken %s seconds' % str(time() - start))

y_prediction = model.predict_class(x_test)
print('\n\naccuracy', np.sum(y_prediction == y_test) / float(len(y_test)))

##########################################


# 2-layer Neural Network
#############################################

start = time()


model = Sequential()
model.add(Dense(input_dim=4, output_dim=100))
model.add(Dense(output=2))
model.add(Activation("softmax"))


mode.compile(loss='categorical_crossentropy',
             optimizer='sgd', metrics=['accuracy'])
mode.fit(x_train, y_train_onehot)

print('\ntime take %s seconds' % str(time() - start))

y_prediction = model.predict_class(x_test)

print('\n\naccuracy', np.sum(y_prediction == y_test) / float(len(y_test)))

##########################################################################


# 3-layer Neural Network
#############################################

start = time()

model = Sequential()
model.add(Dense(input_dim=4, output_dim=100))
model.add(Dense(output=100))
model.add(Dense(output=2))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])


model.fit(X_train, y_train_onehot)

print('\ntime taken %s seconds' % str(time() - start))

y_prediction = model.predict_class(x_test)

print('\n\naccuracy', np.sum(y_prediction == y_test) / float(len(y_test))

##########################################################################
