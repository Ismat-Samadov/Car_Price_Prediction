from preprocessor import preprocess_data,original_features
import joblib
import pandas as pd

def preprocess_input_data(input_data):
    # Apply the same preprocessing steps used during training
    input_data = preprocess_data(input_data)
    # Extract the features used during training
    input_features = input_data[original_features]
    return input_features

if __name__ == "__main__":
    # Load the trained model from the pickle file
    model_path = 'xgboost_model.pkl'
    loaded_model = joblib.load(model_path)

    # Example new data point for prediction (replace this with your actual data)
    new_data = {
        'city': 'Baku',
        'make': 'Toyota',
        'model': 'Camry',
        'year': 2019,
        'ban_type': 'Sedan',
        'colour': 'Silver',
        'engine': 150.0,
        'ride_km': 50000.0,
        'transmission': 'Automatic',
        'gear': 'Front',
        'is_new': 'No'
    }

    # Convert the new data point to a DataFrame
    input_data = pd.DataFrame([new_data])

    # Preprocess the input data
    input_features = preprocess_input_data(input_data)

    # Make predictions using the loaded model
    predicted_price = loaded_model.predict(input_features)

    print(f"Predicted Car Price: {predicted_price[0]:.2f} AZN")

