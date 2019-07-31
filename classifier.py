import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

kf = KFold(n_splits=10, shuffle=True)


def calc(dataset):
    with open('./' + dataset + '/weights.txt', 'rb') as f:
        X = pickle.load(f, encoding='latin1')

    with open('./' + dataset + '/' + dataset + '_graph_labels.txt', 'r') as f:
        label = f.readlines()

    y = [1 if x.strip() == '1' else 0 for x in label]

    acc = []

    ind = list(range(len(y)))

    for train, test in kf.split(ind):
        X_train = np.array([X[i, :] for i in train])
        y_train = np.array([y[i] for i in train])
        X_test = np.array([X[i, :] for i in test])
        y_test = np.array([y[i] for i in test])

        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)

        y_pred = list(clf.predict(X_test))

        acc.append(np.sum([1 if y_pred[i] == y_test[i] else 0 for i in range(len(y_test))]) / len(y_test))

    return acc


acc = calc('NCI1')

print(np.mean(acc))
print(np.std(acc))
