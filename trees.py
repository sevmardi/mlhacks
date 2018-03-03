from math import log


def calcShannonEnt(dataSet):
    """
    Function to calculate the Shannon entropy of a dataset
    """
    num_entries = len(dataSet)
    label_counts = {}
    for fetVec in dataSet:
        current_label = fetVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
            label_counts[current_label] += 1
    shannonEnt = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def create_dataset():
    """The dataset itself"""
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']

    return data_set, labels


def split_dataset(dataset, axis, value):
    """
    DataSet splitting on a given feature
    """
    pass
