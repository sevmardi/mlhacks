from sklearn.datasets import load_digits
import matplotlib
import matplotlib.pyplot as plt 
digits = load_digits()
X, y= digits["data"], digits["target"]
# print(y.shape)
# some_digit= X[1600]
# some_digit_image = some_digit.reshape(64,64)
# plt.imshow(some_digit,cmap=matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone 

skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X, y):
    clone_clf = clone(sgd_clf)
    x_train_folds = X_train[train_index]
    y_train_folds = y_train[test_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train[test_index]

    clone_clf.fit(x_train_folds,y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct=sum(y_pred==y_test_folds)

from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X),1),dtype=bool)
    


    