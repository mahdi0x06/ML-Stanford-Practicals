import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    best_mse = np.inf
    best_tau = tau_values[0]
    
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau=tau) 
        model.fit(x_train, y_train)
        y_predict_valid = model.predict(x_valid)
        mse = np.mean((y_valid - y_predict_valid) ** 2)
        
        # Plot data (for each tau on validation set)
        plt.figure()
        plt.plot(x_train[:, 1], y_train, 'bx', label='Train')
        plt.plot(x_valid[:, 1], y_predict_valid, 'ro', label='Validation Predict')
        plt.legend()
        plt.savefig(f"output/p05c_plot_tau_{tau}.pdf")
        plt.close()
        
        if mse < best_mse:
            best_tau = tau
            best_mse = mse

    # Fit a LWR model with the best tau value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    best_model = LocallyWeightedLinearRegression(tau=best_tau)
    best_model.fit(x_train, y_train)
    
    # Run on the test set to get the MSE value
    y_predict_test = best_model.predict(x_test)
    test_mse = np.mean((y_test - y_predict_test) ** 2)

    # Save predictions to pred_path
    np.savetxt(pred_path, y_predict_test)
    
    # Plot data (Optional: final plot for the test set)
    plt.figure()
    plt.plot(x_train[:, 1], y_train, 'bx', label='Train')
    plt.plot(x_test[:, 1], y_predict_test, 'ro', label='Test Predict')
    plt.legend()
    plt.savefig(f"output/p05c_plot_test.pdf")
    plt.close()
    # *** END CODE HERE ***