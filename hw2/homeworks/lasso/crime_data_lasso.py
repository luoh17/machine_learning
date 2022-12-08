if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd


from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    #raise NotImplementedError("Your Code Goes Here")
    X_train = df_train.drop('ViolentCrimesPerPop', axis=1).to_numpy()
    y_train = df_train['ViolentCrimesPerPop'].to_numpy()

    X_test = df_test.drop('ViolentCrimesPerPop', axis=1).to_numpy()
    y_test = df_test['ViolentCrimesPerPop'].to_numpy()


    #generate lambdas
    lambda_max = np.max(np.sum(2*X_train*(y_train - np.mean(y_train))[:, None], axis =0))
    lambdas = [lambda_max / (2**i) for i in range(17)]

    #save results
    non_zero = []
    train_mse = []
    test_mse = []
    w_path = []

    #make predictions
    def predict(X, w, b):
        return X.dot(w) + b
    
    def mse(x,y):
        return np.mean((x-y) ** 2)

    w_train, bias = train(X_train, y_train, 30)
    for k in lambdas:
        train_w, bias = train(X_train,y_train,k)
        res_non_zero = np.sum(abs(train_w)>1e-10)
        non_zero.append(res_non_zero)

        w_path.append(np.copy(train_w))

        train_mse. append(mse(predict(X_train, train_w, bias), y_train))
        test_mse.append(mse(predict(X_test, train_w, bias), y_test))

    #generate plots
    plt.figure(figsize=(12,8))
    plt.plot(lambdas, non_zero, '--')
    plt.xscale('log')
    plt.title('number of nonzero counts')
    plt.xlabel('log(lambda)')
    plt.ylabel('number of nonzero variables')
    plt.show()


    #generate plots
    plt.figure(figsize=(12,8))
    w_path = np.array(w_path)
    col_names = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65Up', 'householdSize']
    col_index = [3,12,7,5,1]
    for p, label in zip(w_path[:, col_index].T, col_names):
        plt.plot(lambdas, p, '-o', label = label)
    plt.legend()
    plt.xscale('log')
    plt.title('Path')
    plt.xlabel('log(lambda)')
    plt.ylabel('weights')
    plt.show()


    #generate plots
    plt.figure(figsize=(12,8))
    plt.plot(lambdas, train_mse, '-o', label = 'train_mse')
    plt.plot(lambdas, test_mse, '-o', label = 'test_mse')
    plt.xscale('log')
    plt.title('train and test mse')
    plt.legend()
    plt.xlabel('log(lambda)')
    plt.ylabel('MSE')
    plt.show()




if __name__ == "__main__":
    main()
