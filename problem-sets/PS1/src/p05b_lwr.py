import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    model = LocallyWeightedLinearRegression(tau=tau)
    model.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_predict = model.predict(x_eval)

    mse = np.mean((y_eval - y_predict) ** 2)

    plt.figure()
    plt.plot(x_train[:, 1], y_train, 'bx', label='Training Set')
    plt.plot(x_eval[:, 1], y_predict, 'ro', label='Validation Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(f'output/p05b_plot_tau_{tau}.pdf')


    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set."""
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m_eval = x.shape[0]
        y_pred = np.zeros(m_eval)
        
        for i in range(m_eval):
            W = np.diag(np.exp(-np.linalg.norm(self.x - x[i], axis=1)**2 / (2 * self.tau**2)))
            theta = np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y
            y_pred[i] = theta.T @ x[i]
            
        return y_pred
        # *** END CODE HERE ***
