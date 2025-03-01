#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import math

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the Diabetes dataset
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # Append a new feature to all input data, with value "1"
    data_w_ones = np.c_[ dataset.data, np.ones(int(dataset.data.size / 10)) ]

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    data_train, data_test, target_train, target_test = sklearn.model_selection.train_test_split(data_w_ones, dataset.target, test_size=args.test_size, random_state=args.seed)

    # Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    weights = np.dot(np.dot(np.linalg.inv(np.dot(data_train.T, data_train)), data_train.T), target_train)

    # Predict target values on the test set.
    prediction = np.dot(data_test, weights)

    # Manually compute root mean square error on the test set predictions.
    np.dot(prediction.T, target_test) - target_test
    sum = 0
    for i in range(0, prediction.size):
        sum += math.pow(prediction[i] - target_test[i], 2)

    sum *= 1/prediction.size

    sum = math.sqrt(sum)

    return sum


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
