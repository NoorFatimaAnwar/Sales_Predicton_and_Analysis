import numpy as np

def predict_sales(tv, radio, newspaper, scaler, model):
    user_data = np.array([[tv, radio, newspaper]])
    user_data_scaled = scaler.transform(user_data)
    prediction = model.predict(user_data_scaled)
    return prediction[0]
