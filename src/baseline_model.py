import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def baseline_model(y_train, y_test):
    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)

    mse = mean_squared_error(y_test, baseline_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, baseline_pred)
    r2 = r2_score(y_test, baseline_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}
