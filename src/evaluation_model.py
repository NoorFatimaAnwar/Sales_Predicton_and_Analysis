import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    metrics = {
        "Train": {
            "MAE": mean_absolute_error(y_train, y_train_pred),
            "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "R2": r2_score(y_train, y_train_pred)
        },
        "Test": {
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "R2": r2_score(y_test, y_test_pred)
        }
    }

    return metrics, y_train_pred, y_test_pred

def plot_predictions(y_train, y_train_pred, y_test, y_test_pred):
    plt.figure(figsize=(8,6))
    plt.subplot(1,2,1)
    plt.scatter(y_train, y_train_pred, color="blue", alpha=0.6)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--")
    plt.title("Train: Actual vs Predicted")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")

    plt.subplot(1,2,2)
    plt.scatter(y_test, y_test_pred, color="green", alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title("Test: Actual vs Predicted")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")

    plt.tight_layout()
    plt.show()
