if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    If the handout instructs you to implement the following sub-problems, you should:
        a. Run Lloyd's Algorithm for k=10, and report 10 centers returned.
        b. For ks: 2, 4, 8, 16, 32, 64 run Lloyd's Algorithm,
            and report objective function value on both training set and test set.
            (All one plot, 2 lines)

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    #raise NotImplementedError("Your Code Goes Here")

    # a
    center_10 = lloyd_algorithm(x_train, 10)
    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5* num_col, 2 * num_row))

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(center_10[0][i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Centroid {i}') 
    plt.show()

    #b
    ks = [2, 4, 8, 16, 32, 64] 
    train_err = [] 
    test_err = []

    for k in ks:
        res = lloyd_algorithm(x_train, k) 
        train_err.append(res[1][-1]) 
        test_err.append(calculate_error(x_test, res[0]))
    
    plt.figure()
    plt.plot(ks, train_err, label='Train')
    plt.plot(ks, test_err, label='Test')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    main()
