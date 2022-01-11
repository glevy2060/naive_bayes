import numpy as np
import matplotlib.pyplot as plt


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


def bayeslearn(x_train: np.array, y_train: np.array):
    all_pos = np.mean(y_train == 1)
    num_of_ones = np.count_nonzero(y_train == 1)
    num_of_neg_ones = np.count_nonzero(y_train == -1)
    ppos = np.zeros(len(x_train[0]))
    pneg = np.zeros(len(x_train[0]))
    for i in range(len(x_train[0])):
        sum_pos = 0
        sum_neg = 0
        for j in range(len(x_train)):
            if x_train[j][i] == 1 and y_train[j] == x_train[j][i]:
                sum_pos += 1
            if x_train[j][i] == 1 and y_train[j] != x_train[j][i]:
                sum_neg += 1
        ppos[i] = sum_pos / num_of_ones
        pneg[i] = sum_neg / num_of_neg_ones

    return all_pos, ppos, pneg


def calc_sum(all_pos, ppos, pneg, x):
    pred = np.log(all_pos / (1-all_pos))
    for i in range(len(x)):
        if x[i] == 1:
            pred += np.log(ppos[i] / pneg[i])
        else:
            pred += np.log((1-ppos[i]) / (1-pneg[i]))
    return pred


def bayespredict(allpos: float, ppos: np.array, pneg: np.array, x_test: np.array):
    y = np.zeros(len(x_test))
    for i in range(len(x_test)):
        pred = calc_sum(allpos, ppos, pneg, x_test[i])
        if pred >= 0:
            y[i] = 1
        else:
            y[i] = -1
    return y.reshape((len(y), 1))



def simple_test():
    # load sample data from question 2, digits 3 and 5 (this is just an example code, don't forget the other part of
    # the question)
    data = np.load('mnist_all.npz')

    train3 = data['train3']
    train5 = data['train5']

    test3 = data['test3']
    test5 = data['test5']

    m = 500
    n = 50
    d = train3.shape[1]

    x_train, y_train = gensmallm([train3, train5], [-1, 1], m)

    x_test, y_test = gensmallm([test3, train5], [-1, 1], n)

    # threshold the images (binarization)
    threshold = 128
    x_train = np.where(x_train > threshold, 1, 0)
    x_test = np.where(x_test > threshold, 1, 0)

    # run naive bayes algorithm
    allpos, ppos, pneg = bayeslearn(x_train, y_train)

    assert isinstance(ppos, np.ndarray) \
           and isinstance(pneg, np.ndarray), "ppos and pneg should be numpy arrays"

    assert 0 <= allpos <= 1, "allpos should be a float between 0 and 1"

    y_predict = bayespredict(allpos, ppos, pneg, x_test)

    assert isinstance(y_predict, np.ndarray), "The output of the function bayespredict should be numpy arrays"
    assert y_predict.shape == (n, 1), f"The output of bayespredict should be of size ({n}, 1)"

    print(f"Prediction error = {np.mean(y_test != y_predict)}")

def q7a():
    X = np.asarray([[1, -2, 5, 4], [3, 2, 1, -5], [-10, 1, -4, 6]])
    A = X.T @ X
    eigenvalues, _ = np.linalg.eig(A)
    sorted_eigen = np.sort(eigenvalues)
    return sorted_eigen[0] + sorted_eigen[1]


def q7b():
    X = np.asarray([[1, -2, 5, 4], [3, 2, 1, -5], [-10, 1, -4, 6]])
    A = X.T @ X
    values, eigenvectors = np.linalg.eig(A)
    a = np.reshape(eigenvectors[:, 0], (4, 1))
    b = np.reshape(eigenvectors[:, 1], (4, 1))
    U = np.append(a, b, axis = 1)
    return U, X


def q7c(U, X):
    x1 = U @ U.T @ X[0]
    x2 = U @ U.T @ X[1]
    x3 = U @ U.T @ X[2]
    re_x = np.asarray([x1, x2, x3])
    return np.linalg.norm(X - re_x) ** 2




if __name__ == '__main__':
    simple_test()
    # print(q7a())
    # U, X = q7b()
    # print(q7c(U, X))
