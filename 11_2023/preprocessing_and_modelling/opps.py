import streamlit as st
import joblib
import pandas as pd
xgboost_model = joblib.load('xgboost_model.pkl')
datacleaned = pd.read_excel('datacleaned.xlsx')

car_image_url = "https://images.unsplash.com/photo-1511919884226-fd3cad34687c?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fGNhcnxlbnwwfHwwfHx8MA%3D%3D"
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
st.sidebar.title("Input Parametres")
st.title("Car Price Prediction App")
city = st.sidebar.selectbox("Select which city you want to buy:",datacleaned['city'].unique())
make = st.sidebar.selectbox("Select which make you want to buy",datacleaned['make'].unique())
model = st.sidebar.selectbox("Select which model you want to buy",datacleaned['model'].unique())
year = st.sidebar.selectbox("Select which year's car you want to buy",datacleaned['year'].unique())
ban_type = st.sidebar.selectbox("Select which ban type car you want to buy",datacleaned['ban_type'].unique())
colour = st.sidebar.selectbox("Select which colour car you want to buy",datacleaned['colour'].unique())
engine_power = st.sidebar.selectbox("Select engine_power of car you want to buy",datacleaned['engine_power'].unique())
ride_km = st.sidebar.slider("Select km of car you want to buy",0,5560000,50)
transmission = st.sidebar.selectbox("Select transmission of car you want to buy", datacleaned['transmission'].unique())
gear = st.sidebar.selectbox("Select gear of car you want to buy", datacleaned['gear'].unique())
is_new = st.sidebar.selectbox("This car is new ?", ("Yes","No"))

data = {
    "city": city,
    "make": make,
    "model": model,
    "year": year,
    "ban_type": ban_type,
    "colour": colour,
    "engine_power": engine_power,
    "ride_km":ride_km,
    "transmission":transmission,
    "gear":gear,
    "is_new":1 if is_new == "Yes" else 0
}

columns = datacleaned.columns
df = pd.DataFrame.from_dict([data])
df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

if st.sidebar.button('Predict Price'):
    prediction = xgboost_model.predict(df)
    st.success(f'Predicted Price: {prediction}')