#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import math

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int, help="Number of SGD training epochs")
parser.add_argument("--l2", default=0.0, type=float, help="L2 regularization strength")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[float], float, float]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate an artificial regression dataset
    data, target = sklearn.datasets.make_regression(n_samples=args.data_size, random_state=args.seed)

    # Append a constant feature with value 1 to the end of every input data
    data_ones = np.vstack([data.T, np.ones(data.shape[1])]).T
    #print(data_ones)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(data_ones, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial linear regression weights
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    train_rmses, test_rmses = [], []
    for epoch in range(args.epochs):
        #print('epoch iteration: ', epoch)
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # A gradient for example `(x_i, t_i)` is `(x_i^T weights - t_i) * x_i`,
        # and the SGD update is
        #   weights = weights - args.learning_rate * (gradient + args.l2 * weights)`.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.

        gradient = 0
        for batch_counter in range(train_data.shape[0] // args.batch_size): #one batch

            batch_sum = 0
            for i in range(args.batch_size):
                idx = batch_counter * args.batch_size + i
                x = permutation[idx]
                batch_sum += np.dot((np.dot(x, weights) - train_target[idx]), x)

            gradient += 1/args.batch_size * batch_sum
        weights = weights - args.learning_rate * (gradient + args.l2 * weights)

        # TODO: Append current RMSE on train/test to `train_rmses`/`test_rmses`.

        train_sum = 0
        for i in range(train_data.shape[0]):
            train_sum = np.power((np.dot(train_data[i], weights) - train_target[i]), 2)
            train_sum *= 1/train_data.shape[0]
            train_sum = math.sqrt(train_sum)

        for i in range(test_data.shape[0]):
            test_sum = np.power((np.dot(test_data[i], weights) - test_target[i]), 2)
            test_sum *= 1/test_data.shape[0]
            test_sum = math.sqrt(test_sum)

        train_rmses.append(train_sum)
        test_rmses.append(test_sum)

        

    # TODO: Compute into `explicit_rmse` test data RMSE when fitting
    # `sklearn.linear_model.LinearRegression` on `train_data` (ignoring `args.l2`).
    sk_model = sklearn.linear_model.LinearRegression().fit(train_data, train_target)
    prediction = sk_model.predict(test_data)

    explicit_rmse = sklearn.metrics.mean_squared_error(test_target, prediction, squared=False)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(train_rmses, label="Train")
        plt.plot(test_rmses, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return weights, test_rmses[-1], explicit_rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, sgd_rmse, explicit_rmse = main(args)
    print("Test RMSE: SGD {:.2f}, explicit {:.2f}".format(sgd_rmse, explicit_rmse))
    print("Learned weights:", *("{:.2f}".format(weight) for weight in weights[:12]), "...")
