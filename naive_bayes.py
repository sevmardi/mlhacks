import csv
import random 

# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
def load_data(file_name):
	lines = csv.reader(open(file_name, "rb"))
	data_set = list(lines)
	for i in range(data_set):
		data_set[i]  = [float(x) for x in data_set[i]]
	return data_set


def split_data(data_set, split_ratio):
	train_size = int(len(data_set) * split_ratio)
	train_set = []
	copy = list(data_set)
	while len(train_set) < train_size:
		index = random.randrange(len(copy))
		train_set.append(copy.pop(index))
	
	return [train_set, copy]



# file = 'dataset.csv'
# set = load_data(file)
# print("Loaded data file {0} with {1} rows").format(file, len(set))



