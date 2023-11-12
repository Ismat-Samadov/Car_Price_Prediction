import joblib
import pandas as pd
import streamlit as st
car_image_url = "https://images.unsplash.com/photo-1603557275022-48fd88ce4933?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8b2xkJTIwY2FyfGVufDB8fDB8fHww"
image_html = f'<img src="{car_image_url}" style="max-width: 100%; height: auto;">'
st.markdown(image_html, unsafe_allow_html=True)
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

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
input_data['city'] = st.sidebar.selectbox('city', city)
input_data['make'] = st.sidebar.selectbox('make', make)
input_data['model'] = st.sidebar.selectbox('model', model)
input_data['year'] = st.sidebar.selectbox('year', year)
input_data['ban_type'] = st.sidebar.selectbox('ban_type', ban_type)
input_data['colour'] = st.sidebar.selectbox('colour', colour)
input_data['engine_power'] = st.sidebar.selectbox('engine_power', engine_power)
input_data['ride_km'] = st.sidebar.selectbox('ride_km', ride_km)
input_data['transmission'] = st.sidebar.selectbox('transmission', transmission)
input_data['gear'] = st.sidebar.selectbox('gear', gear)
input_data['is_new'] = st.sidebar.selectbox('is_new', is_new)
input_df = pd.DataFrame(input_data, index=[0])

dummy_cols = pd.get_dummies(preprocessed_data, drop_first=True)
input_df = pd.get_dummies(input_df, drop_first=True).reindex(columns=dummy_cols.columns, fill_value=0)

if 'model' in locals():
    # Add a button to trigger model prediction
    if st.sidebar.button('Predict Car Price'):
        prediction = xgboost_model.predict(input_df)
        st.balloons()
        st.sidebar.success(f"Predicted Car Price: {prediction[0]:,.2f} AZN")
else:
    st.warning("Model not loaded. Please check the model file path.")
