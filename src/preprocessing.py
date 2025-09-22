import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filename="Advertising.csv"):
    base_dir = os.path.dirname(os.path.dirname(__file__))   # go to project root
    path = os.path.join(base_dir, "data", filename)
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df

def split_and_scale(df):
    X = df[["TV", "Radio", "Newspaper"]]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
