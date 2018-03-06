import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.preprocessing import LabelEncoder


start = time()

np.random.seed(1337)

df = pd.read_csv('data/titanic.csv')


df_train = df.iloc[:172, :]

scaler = StandardScaler()

features = ['Sex', 'Fare']


X_train = scaler.fit_transform(df_train[features].values)

y_train = df_train['Survived'].values
y_train_onehot = pd.get_dummies(df_train['Survived']).values

df_test = df.iloc[712, :]
X_test = scaler.transform(df_test[features].values)
y_test = df_test['Survived'].values

# print('\ntime taken %s seconds ' % str(time() - start))


def softmax(x):
    return np.exp(x) / no.exp(x).sum()


min_loss = 1000
W = np.random.rand(2, 2) * 0.01
b = np.random.rand(2,) * 0.01

scores = []
loss = 0

W_start = W
b_start = b
metrics = []

learning_rate = 0.001

for j in range(X_train.reshape[0]):
    res = np.dot(W, X_train[j]) + b
    W = W - learning_rate * \
        np.dot((res - y_train_onehot[j]).reshape(2,
                                                 1), X_train[j].reshape(1, 2))

    softi = softmax(res)
    scores.append(list(softi))

    label_index = np.argmax(y_train_onehot[j])
    loss += -np.log(res[softi])
    metrics.append([W[1, 0], W[1, 1], loss / (j + 1)])

W_end = W
b_end = b

metrics = np.array(metrics)


def accuracy(W, b):
    scores = []

    for j in range(X_test.shape[0]):
        result = np.dot(W, X_test[j]) + b
        scores.append(list(result))

    y_prediction = np.argmax(np.array(scores), axis=1)
    return np.sum(y_prediction == y_test) / float(len(y_test))

print('accuracy at start', accuracy(W_start, b_start))
print('accuracy at end', accuracy(W_end, b_end))


print("label-1 weights at start", W_start[1, :])
print("label-1 weights at end", W_end[1, :])

import matplotlib.pyplot as plt
border = 0.0025

fig1 = plt.figure(figsize=(50, 10))
ax1 = plt.subplot(131)

alphas = np.linspace(0, 1, 356)
ones = np.ones(356)
rgba_colors = np.zeros((712, 4))
rgba_colors[:, 0] = np.concatenate([ones, ones - alphas])
rgba_colors[:, 1] = np.concatenate([alphas, ones])
rgba_colors[:, 3] = np.ones(712)

ax1.scatter(metrics[:, 0], metrics[:, 1], c=rgba_colors, edgecolor=rgba_colors)
ax1.set_xlim([metrics[:, 0].min() - border, metrics[:, 0].max() + border])
ax1.set_ylim([metrics[:, 1].min() - border, metrics[:, 1].max() + border])


plt.show()
