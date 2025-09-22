from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_models(X_train_scaled, y_train, X_test_scaled, y_test):
    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results["Linear Regression"] = {
        "MAE": mean_absolute_error(y_test, y_pred_lr),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        "R2": r2_score(y_test, y_pred_lr)
    }

    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    results["Ridge Regression"] = {
        "MAE": mean_absolute_error(y_test, y_pred_ridge),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
        "R2": r2_score(y_test, y_pred_ridge)
    }

    # SVM
    svm = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    results["SVM"] = {
        "MAE": mean_absolute_error(y_test, y_pred_svm),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_svm)),
        "R2": r2_score(y_test, y_pred_svm)
    }

    return results, lr, ridge, svm
