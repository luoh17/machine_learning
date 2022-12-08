"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple
from typing_extensions import Self

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight = None 
        # You can add additional fields
        #raise NotImplementedError("Your Code Goes Here")
        self.mean = None
        self.std = None
        


    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        #raise NotImplementedError("Your Code Goes Here")
        new_arr = np.zeros(shape = (len(X), degree))

        for i in range(len(X)):
            for j in range(degree):
                new_arr[i][j] = X[i] ** (j+1)
        
        
        return new_arr


    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        #raise NotImplementedError("Your Code Goes Here")
        
        n = len(X)
        # expand X to be a n*d array with degree d
        new_feat = self.polyfeatures(X, self.degree)

        # compute the statistics for train and test data
        if n != 1:
            self.mean = new_feat.mean(0)
            self.std = new_feat.std(0) 
        else:
            self.mean = 0
            self.std = 1

        new_feat = (new_feat-self.mean)/self.std

        # add the feature row with order 0
        my_feat = np.c_[np.ones([n,1]), new_feat]
        
    
        # construct reg matrix
        regMatrix = self.reg_lambda * np.eye(self.degree + 1)
        regMatrix[0,0]=0

        #  solution 
        self.weight = np.linalg.pinv(my_feat.T.dot(my_feat) + regMatrix).dot(my_feat.T).dot(y)


    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        #raise NotImplementedError("Your Code Goes Here")
        n= len(X)
        # expand X to be a n*d array with degree d
        new_feat = self.polyfeatures(X, self.degree)


        # compute the statistics for train and test data
        new_feat = (new_feat-self.mean)/self.std

        # add the feature row with order 0
        my_feat = np.c_[np.ones([n,1]), new_feat]

        # predict
        return my_feat.dot(self.weight)
    



@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    #raise NotImplementedError("Your Code Goes Here")
    mse = sum((a-b)**2)/len(a)
    return mse


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    #raise NotImplementedError("Your Code Goes Here")
    

    for i in range(1, n):
        xtrain = Xtrain[0:i+1]
        ytrain = Ytrain[0:i+1]
        model = PolynomialRegression(degree = degree, reg_lambda = reg_lambda)
        model.fit(xtrain,ytrain)

        pred_train = model.predict(xtrain)
        pred_test = model.predict(Xtest)
        errorTrain[i] = np.mean((pred_train -ytrain)**2)
        errorTest[i] =  np.mean((pred_test -Ytest)**2)

    return (errorTrain, errorTest)
