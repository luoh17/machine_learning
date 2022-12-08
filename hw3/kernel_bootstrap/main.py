from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem
RNG = np.random.RandomState(seed=2022)


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    #raise NotImplementedError("Your Code Goes Here")
    return (1+np.multiply.outer(x_i,x_j))**d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    #raise NotImplementedError("Your Code Goes Here")
    return np.exp(-gamma*np.subtract.outer(x_i,x_j)**2)


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    #raise NotImplementedError("Your Code Goes Here")
    X_mean, X_std = np.mean(x), np.std(x)
    x = (x - X_mean) / X_std
    
    K = kernel_function(x,x,kernel_param)
    return np.linalg.solve(K + _lambda*np.eye(K.shape[0]),y)


def predict(x_train, x_val, alpha, kernel_function, kernel_param):
    X_mean, X_std = np.mean(x_train), np.std(x_train)
    x_train = (x_train - X_mean) / X_std
    x_val = (x_val - X_mean) / X_std
    return alpha@kernel_function(x_train, x_val, kernel_param)
    

@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    #raise NotImplementedError("Your Code Goes Here")
    myfolds = RNG.choice((list(np.arange(num_folds))*(fold_size))[:len(x)]
    ,size=len(x), replace = False)
    err = np.zeros(num_folds)

    
    for i in range(num_folds):
        idx = (myfolds != i)
        alpha = train(x[idx], y[idx], kernel_function, kernel_param, _lambda)
        y_pred_val = predict(x[idx],x[~idx],alpha, kernel_function, kernel_param)
        err[i] = np.mean((y[~idx] - y_pred_val)**2)
    return np.mean(err)



@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2 for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    #raise NotImplementedError("Your Code Goes Here")
    lambdas = ([10 ** -i for i in range(1,6)])
    gamma = 1/ np.median(np.subtract.outer(x,x) ** 2)
    berr = np.inf
    be1 = None
    beg = None

    for k in lambdas:
        e = cross_validation(x,y,rbf_kernel,gamma,1,num_folds)
        if e < berr:
            be1 = k
            beg = gamma
            berr = e 
    return be1, beg


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You do not really need to search over gamma. 1 / median((x_i - x_j) for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """
    #raise NotImplementedError("Your Code Goes Here")
    lambdas = ([10 ** -i for i in range(1,6)])
    ds = [i for i in range(7,22)]
    berr = np.inf
    be1 = None
    bed = None

    for k in lambdas:
        for d in ds:
            e = cross_validation(x,y,poly_kernel,d,1,num_folds)
            if e < berr:
                be1 = k
                bed = d
                berr = e 
    return be1, bed


@problem.tag("hw3-A", start_line=1)
def bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    bootstrap_iters: int = 300,
) -> np.ndarray:
    """Bootstrap function simulation empirical confidence interval of function class.

    For each iteration of bootstrap:
        1. Sample len(x) many of (x, y) pairs with replacement
        2. Train model on these sampled points
        3. Predict values on x_fine_grid (see provided code)

    Lastly after all iterations, calculated 5th and 95th percentiles of predictions for each point in x_fine_point and return them.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        bootstrap_iters (int, optional): [description]. Defaults to 300.

    Returns:
        np.ndarray: A (2, 100) numpy array, where each row contains 5 and 95 percentile of function prediction at corresponding point of x_fine_grid.

    Note:
        - See np.percentile function.
            It can take two percentiles at the same time, and take percentiles along specific axis.
    """
    x_fine_grid = np.linspace(0, 1, 100)
    #raise NotImplementedError("Your Code Goes Here")
    preds = []
    indices = np.arange(x.shape[0])

    for i in range(bootstrap_iters):
        boot_idx = np.random.choice(indices, size=x.shape[0], replace=True)
        X_boot, Y_boot = x[boot_idx], y[boot_idx]
        alpha = train(X_boot, Y_boot, kernel_function, kernel_param, _lambda)
        
        preds.append([predict(X_boot, x_fine_grid, alpha, kernel_function, kernel_param)])
        
        CI_low = np.percentile(preds, 5, axis = 0)
        CI_up = np.percentile(preds, 95, axis = 0)
    return np.concatenate((CI_low, CI_up), axis = 0)


@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid
        C. For both rbf and poly kernels, plot 5th and 95th percentiles from bootstrap using x_30, y_30 (using the same fine grid as in part B)
        D. Repeat A, B, C with x_300, y_300
        E. Compare rbf and poly kernels using bootstrap as described in the pdf. Report 5 and 95 percentiles in errors of each function.

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    #raise NotImplementedError("Your Code Goes Here")
    # question a ++++++++++++++++++++++++++++++++++++
    poly_l, poly_d = poly_param_search(x_30, y_30,30)
    print(f"optimual lambda and degree for poly kernel are {poly_l}, {poly_d}")

    rbf_l, rbf_d = rbf_param_search(x_30,y_30,30)
    print(f"optimual lambda and degree for rbf kernel are {rbf_l}, {rbf_d}")

    # question b ++++++++++++++++++++++++++++++++++++
    trueX = np.linspace(0,1,100)
    trueY = f_true(trueX)

    poly_a = train(x_30,y_30,poly_kernel,10,0.001 )
    poly_pred_y = predict(x_30, sorted(x_30), poly_a, poly_kernel, 10)

    plt.scatter(x_30,y_30)
    plt.plot(sorted(x_30), poly_pred_y, label = "polynomial predictions", color = 'red')
    plt.plot(trueX,trueY,label = "truth", color = 'blue')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('polynomial kernel with n = 30')
    plt.show()


    rbf_a = train(x_30,y_30,rbf_kernel,11.2,0.1 )
    rbf_pred_y = predict(x_30, sorted(x_30), rbf_a, rbf_kernel, 11.2)

    plt.scatter(x_30,y_30)
    plt.plot(sorted(x_30), rbf_pred_y, label = "rbf predictions", color = 'green')
    plt.plot(trueX,trueY,label = "truth", color = 'blue')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('rbf kernel with n = 30')
    plt.show()


    # question c ++++++++++++++++++++++++++++++++++++
    poly_l, poly_d = poly_param_search(x_300, y_300,10)
    print(f"optimual lambda and degree for poly kernel are {poly_l}, {poly_d}")

    rbf_l, rbf_d = rbf_param_search(x_300,y_300,10)
    print(f"optimual lambda and degree for rbf kernel are {rbf_l}, {rbf_d}")

    trueX = np.linspace(0,1,100)
    trueY = f_true(trueX)

    poly_a = train(x_300,y_300,poly_kernel,14,0.1 )
    poly_pred_y = predict(x_300, sorted(x_300), poly_a, poly_kernel, 14)

    plt.scatter(x_300,y_300)
    plt.plot(sorted(x_300), poly_pred_y, label = "polynomial predictions", color = 'red')
    plt.plot(trueX,trueY,label = "truth", color = 'blue')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('polynomial kernel with n = 300')
    plt.show()


    rbf_a = train(x_300,y_300,rbf_kernel,12.87,0.1 )
    rbf_pred_y = predict(x_300, sorted(x_300), rbf_a, rbf_kernel, 12.87)

    plt.scatter(x_300,y_300)
    plt.plot(sorted(x_300), rbf_pred_y, label = "rbf predictions", color = 'green')
    plt.plot(trueX,trueY,label = "truth", color = 'blue')

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('rbf kernel with n = 300')
    plt.show()



if __name__ == "__main__":
    main()
