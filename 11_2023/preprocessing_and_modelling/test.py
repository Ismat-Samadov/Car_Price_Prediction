import joblib
import pandas as pd
import streamlit as st

try:
    xgboost_model = joblib.load('xgboost_model.pkl')
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

preprocessed_data = pd.read_excel('preprocess_data.xlsx')
city = sorted(preprocessed_data['city'].unique())
make = sorted(preprocessed_data['make'].unique())
model = sorted(preprocessed_data['model'].unique())
year = sorted(preprocessed_data['year'].unique())
ban_type = sorted(preprocessed_data['ban_type'].unique())
colour = sorted(preprocessed_data['colour'].unique())
engine_power = sorted(preprocessed_data['engine_power'].unique())
ride_km = sorted(preprocessed_data['ride_km'].unique())
transmission = sorted(preprocessed_data['transmission'].unique())
gear = sorted(preprocessed_data['gear'].unique())
is_new = sorted(preprocessed_data['is_new'].unique())

input_data = {}
input_data['city'] = st.selectbox('city', city)
input_data['make'] = st.selectbox('make', make)
input_data['model'] = st.selectbox('model', model)
input_data['year'] = st.selectbox('year', year)
input_data['ban_type'] = st.selectbox('ban_type', ban_type)
input_data['colour'] = st.selectbox('colour', colour)
input_data['engine_power'] = st.selectbox('engine_power', engine_power)
input_data['ride_km'] = st.selectbox('ride_km', ride_km)
input_data['transmission'] = st.selectbox('transmission', transmission)
input_data['gear'] = st.selectbox('gear', gear)
input_data['is_new'] = st.selectbox('is_new', is_new)
input_df = pd.DataFrame(input_data, index=[0])

# Transform user input into dummy variables
dummy_cols = pd.get_dummies(preprocessed_data, drop_first=True)
input_df = pd.get_dummies(input_df, drop_first=True).reindex(columns=dummy_cols.columns, fill_value=0)

# Model prediction section
if 'model' in locals():
    # Add a button to trigger model prediction
    if st.button('Predict Car Price'):
        # Make predictions using the loaded model
        prediction = xgboost_model.predict(input_df)

        # Display the prediction result
        st.success(f"Predicted Car Price: {prediction[0]:,.2f} AZN")
else:
    st.warning("Model not loaded. Please check the model file path.")

# ... (continue with the rest of your Streamlit app)
