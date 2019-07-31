import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=10, shuffle=True)

dataset = 'NCI109'

def calc(dataset):
    with open('./' + dataset + '/weights.txt', 'rb') as f:
        X = pickle.load(f)

    with open('./' + dataset + '/' + dataset + '_graph_labels.txt', 'r') as f:
        label = f.readlines()

    y = [1 if x.strip() == '1' else 0 for x in label]

    acc_train = []
    acc_test = []

    ind = list(range(len(y)))

    for train, test in kf.split(ind):
        X_train = np.array([X[i, :] for i in train])
        y_train = np.array([y[i] for i in train])
        X_test = np.array([X[i, :] for i in test])
        y_test = np.array([y[i] for i in test])

        clf = LogisticRegression(C = 0.25).fit(X_train, y_train)

        y_pred = list(clf.predict(X_test))

        acc_test.append(np.sum([1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_test))]) / len(y_test))
        
        y_pred = list(clf.predict(X_train))

        acc_train.append(np.sum([1 if y_pred[i] == y_train[i] else 0 for i in range(len(y_train))]) / len(y_train))

    return acc_train, acc_test

dataset = 'NCI109'
acc_train, acc_test = calc(dataset)

with open('./' + dataset + '/iter_loss.json', 'rb') as f:
    iter_loss = pickle.load(f)

print(np.mean(acc_train))
print(np.std(acc_train))

print(np.mean(acc_test))
print(np.std(acc_test))
