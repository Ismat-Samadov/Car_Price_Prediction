import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the pre-trained XGBoost model
model = joblib.load('xgboost_model.pkl')

categorical_columns = ['city', 'make', 'model', 'ban_type', 'colour', 'transmission', 'gear', 'is_new']
numeric_columns = ['year', 'engine_power', 'ride_km']
encoder = OneHotEncoder()

# Load your training data (assuming you have it saved or accessible)
training_data = pd.read_excel('datacleaned.xlsx')  # Replace with your actual file path

# Fit the OneHotEncoder on the training data
encoder.fit(training_data[categorical_columns])

def preprocess_input(data):
    categorical_data = data[categorical_columns]
    categorical_encoded = encoder.transform(categorical_data).toarray()
    numeric_data = data[numeric_columns]
    processed_data = pd.concat([pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_columns)),
                               numeric_data], axis=1)

    return processed_data

def predict_price(features):
    processed_features = preprocess_input(features)
    prediction = model.predict(processed_features)
    return prediction

def main():
    st.title("Car Price Prediction App")
    st.sidebar.header("Input Features")
    city = st.sidebar.text_input("City", "")
    make = st.sidebar.text_input("Make", "")
    model_input = st.sidebar.text_input("Model", "")
    year = st.sidebar.number_input("Year", 1940, 2023, step=1)
    ban_type = st.sidebar.selectbox("Body Type",
                                    ["SUV", "Sedan", "Hatchback", "Station Wagon", "Liftback", "Truck", "Van",
                                     "Minivan", "Coupe", "Motorcycle", "Pickup", "Convertible", "Microbus", "Moped",
                                     "Bus", "Roadster", "Quad Bike"])
    colour = st.sidebar.selectbox("Color",
                                  ["Silver", "Black", "Blue", "Gray", "Brown", "Dark_Red", "Red", "Green", "Blue",
                                   "Beige", "Gold", "Brown", "Orange", "Yellow", "Purple", "Pink"])
    engine_power = st.sidebar.number_input("Engine Power", min_value=0.0)
    ride_km = st.sidebar.number_input("Ride (km)", min_value=0.0)
    transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual", "CVT", "Automated_Manual"])
    gear = st.sidebar.selectbox("Gear", ["Full", "Front", "Rear"])
    is_new = st.sidebar.selectbox("Is New", ["Yes", "No"])

    user_inputs = {'city': city, 'make': make, 'model': model_input, 'year': year, 'ban_type': ban_type,
                   'colour': colour,
                   'engine_power': engine_power, 'ride_km': ride_km, 'transmission': transmission, 'gear': gear,
                   'is_new': is_new}
    input_df = pd.DataFrame([user_inputs])
    processed_input = preprocess_input(input_df)
    st.write("Processed Input Data:")
    st.write(processed_input)

    if st.button("Predict Car Price"):
        prediction = predict_price(processed_input)
        st.success(f"Predicted Car Price: {prediction[0]:,.2f} AZN")

        # Display additional information or visualization about the prediction
        # Example: st.line_chart(prediction_history) or st.plotly_chart(prediction_chart)


if __name__ == '__main__':
    main()
