import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***

    # --- Part (c) ---
    # loading the train set
    x_train_c, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)

    # creating an instance and training
    model_c = LogisticRegression()
    model_c.fit(x_train_c, t_train)

    # loading the test set
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    
    # prediction and saving
    predict_c = model_c.predict(x_test)
    np.savetxt(pred_path_c, predict_c)
    
    # Plotting Part (c)
    util.plot(x_test, t_test, model_c.theta, "{}.pdf".format(pred_path_c))


    # --- Part (d) ---
    # loading the train set
    x_train_d, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)

    # creating an instance and training
    model_d = LogisticRegression()
    model_d.fit(x_train_d, y_train)

    # prediction and saving
    predict_d = model_d.predict(x_test)
    np.savetxt(pred_path_d, predict_d)
    
    # Plotting Part (d)
    util.plot(x_test, t_test, model_d.theta, "{}.pdf".format(pred_path_d))


    # --- Part (e) ---
    # Apply correction factor using validation set
    x_valid, y_valid = util.load_dataset(valid_path, label_col="y", add_intercept=True)
    x_valid_pos = x_valid[y_valid == 1]

    # Calculate alpha using model_d
    alpha = np.mean(model_d.predict(x_valid_pos))
    
    # Correct predictions and save
    predict_e = predict_d / alpha
    np.savetxt(pred_path_e, predict_e)
    
    # Plotting Part (e) with correction factor
    util.plot(x_test, t_test, model_d.theta, "{}.pdf".format(pred_path_e), correction=alpha)

    # *** END CODE HERE ***
