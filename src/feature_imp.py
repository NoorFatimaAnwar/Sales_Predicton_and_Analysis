import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def get_feature_importance(model, X_test_scaled, y_test, feature_names):
    perm_importance = permutation_importance(
        model, X_test_scaled, y_test, n_repeats=30, random_state=42
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": perm_importance.importances_mean
    }).sort_values(by="Importance", ascending=False)

    return importance_df

def plot_feature_importance(importance_df):
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importance (Permutation)")
    plt.show()
