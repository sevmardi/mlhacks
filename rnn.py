# Recurrent Neural Network
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from time import time


np.random.seed(1337)

df = pd.read_csv('data/rottentomatoes.csv')

df['Phrase'].values[0]

count = CountVectorizer(analyzer='word')

df_train = df.iloc[:124800, :]

X_train = count.fit_transform(df_train['Phrase'])
y_train = df_train['Sentiment'].values
y_train_onehot = pd.get_dummies(df_train['Sentiment']).values


df_test = df.iloc[124800:, :]
X_test = count.transform(df_test['Phrase'])
y_test = df_test['Sentiment'].values


# for i in range(10):
# 	print(i+250, count.get_feature_names()[i+250])

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=0, verbose=3)
model = model.fit(X_train, y_train)

y_prediction = model.predict(X_test)


print("\naccyracy ", np.sum(y_prediction == y_test) / float(len(y_test)))



from collections import defaultdict

word_to_index = defaultdict(int)

for i, item in enumerate(count.get_feature_names):
	word_to_index[item] = i+1

seq = count.build_analyzer()


def sentence_to_indices(sentence):
	return [word_to_index[word] for word in seq(sentence)]

X_train_seq = map(sentence_to_indices, df_train['Phrase'])
X_test_seq = map(sentence_to_indices, df_test['Phrase'])

from keras.preprocessing import sequence

X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=48)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=48)

df_train['Phrase'].values[0]
seq(df_train['Phrase'].values[0])

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM

start = time()

model = Sequential()
model.add(Embedding(len(word_to_index)+1, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
mode.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train_pad, y_train_onehot, nb_epoch=2)

print('\ntime taken %s seconds ' % str(time() - start)

y_prediction = model.predict_classes(X_test_pad)
print('\naccuracy', np.sum(y_predictio == y_test) / float(len(y_test)))










