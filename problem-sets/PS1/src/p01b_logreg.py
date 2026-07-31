import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    #creating an instance
    model = LogisticRegression()
    model.fit(x_train, y_train)

    #loading the evaluation dataset
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    pred = model.predict(x_eval)

    #saving the prediction 
    np.savetxt(pred_path, pred)

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        
        for _ in range(self.max_iter):
            #constructing the sigmoid function
            z = x @ self.theta
            h = 1 / (1 + np.exp(-z))

            #constructing the gradient of l(theta)
            gradient = (x.T @ (h - y)) / x.shape[0]

            #constructing the hessian matrix
            S = np.diag(h * (1 - h))
            H = (x.T @ S @ x) / x.shape[0]

            #solving the equation (H @ delta = gradient(l))
            delta = np.linalg.solve(H, gradient)

            new_theta = self.theta - delta 

            #checking if we are close enough
            if np.linalg.norm(self.theta - new_theta, ord=1) < self.eps:
                self.theta = new_theta
                break
            self.theta = new_theta
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x @ self.theta))
        # *** END CODE HERE ***
