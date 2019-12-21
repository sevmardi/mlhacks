import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
print('Importing data ...')
import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
safe_print("Loading dataset...")
# mnist = fetch_openml('mnist_784')

# mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='data/')

# mnist = fetch_mldata('MNIST original')
# data, target = pd.read_csv('data/mnist.csv')
# mnist = load_digits()

# x = mnist.data
# y = mnist.target
# print(data.shape)
# print('Data prep ...')
# train_x, test_x, train_y, test_y = train_test_split(x, y,test_size=0.20,random_state=42)


# from sklearn.ensemble import RandomForestClassifier
# rf1 = RandomForestClassifier(max_depth=4,n_estimators=10)
# rf1.fit(train_x, train_y)

# rf3 = RandomForestClassifier(max_depth=16,n_estimators=10)
# rf3.fit(train_x, train_y)

# from PIL import Image
# from scipy.misc import imresize 


# def rescale_strech_image(image):
#     image_data = np.asarray(image)
#     image_data_bw = np.reshape(image_data, (28, 28))
#     non_empty_columns = np.where(image_data_bw.max(axis=0) > 128)[0]
#     non_empty_rows = np.where(image_data_bw.max(axis=1) > 128)[0]
#     cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
#     #print(cropBox)
#     image_data_new = image_data_bw[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1]
#     #image_data_new = np.resize(image_data_new, (20, 20))
#     image_data_new = imresize(image_data_new, (20, 20))
#     #image_data_new.show()
#     return (np.array(image_data_new).astype(np.uint8))

# train_modified = np.apply_along_axis(rescale_strech_image, axis=1, arr=train_x)
# test_modified = np.apply_along_axis(rescale_strech_image, axis=1, arr=test_x)

# train_final = np.reshape(train_modified, (train_modified.shape[0], 400))
# test_final = np.reshape(test_modified, (test_modified.shape[0], 400))