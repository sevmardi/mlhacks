# import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Classify Manhattan with TensorFlow
# we will use TensorFlow to train a neural network to predict whether a
# geolocation is in Manhattan or not, by looking at its longitude and
# latitude.
is_mt = nyc_cols[:, 0].astype(np.int32)  # read the 0th column (is_mt) as int32
# read the 1st and 2nd column (latitude and longitude) as float32
latlng = nyc_cols[:, 1:3].astype(np.float32)
print("Is Manhattan: " + str(is_mt))
print("\nLat/Lng: \n\n" + str(latlng))


lastig = StandardScaler().fit_transform(latlng)
print(np.mean(latlng_std[:, 0]))
print(np.std(latlng_std[:, 0]))
print(np.mean(latlng_std[:, 1]))
print(np.std(latlng_std[:, 1]))


import matplotlib.pyplot as plt
lat = latlng_std[:, 0]
lng = latlng_std[:, 1]
# plot points in Manhattan in blue
plt.scatter(lng[is_mt == 1], lat[is_mt == 1], c='b')
# plot points outside Manhattan in yellow
plt.scatter(lng[is_mt == 0], lat[is_mt == 0], c='y')
plt.show()

# 8,000 pairs for training
latlng_training = latlng_std[0:8000]
is_mt_train = is_mt[0:8000]

# 2,000 pairs for test
latlng_test = latlng_std[8000:10000]
is_mt_test = is_mt[8000:10000]

print('Split Finshed! ')

tf.logging.set_verbosity(tf.logging.ERROR)  # supress warning messages

# define two feature columns with real values
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]


dunc = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns, hidden_units=[], n_classes=2)

# Check the accuracy of the neural network


def plot_predicted_map():
    # an array of prediction results
    is_mt_pred = dunc.predict(latlng_std,  as_iterable=False)

    plt.scatter(lng[is_mt_pred == 1], lat[is_mt_pred == 1], c='b')
    plt.scatter(lng[is_mt_pred == 0], lat[is_mt_pred == 0], c='y')
    plt.show()

def print_accuracy():
	accuracy = dnnc.evaluate(x=latlng_test, y=is_mt_test)["accuracy"]
	print('Accuracy: {:.2%}'.format(accuracy))

dnnc.fit(x=latlng_train, y=is_mt_train, steps=1)
plot_predicted_map()
print_accuracy()

steps = 100
for i in range(1,6):
	dunc.fit(x=latlng_training, y=is_mt_train, steps=steps)
	plot_predicted_map()
	print("steps: ", str(i * steps))
	print_accuracy()

print("\nTraining Finished")



