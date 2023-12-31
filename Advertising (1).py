import streamlit as st
import pandas as pd
import pickle

st.write("""
# Advertising Sales Prediction App

This app predicts the **Advertising Sales** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 231.1, 44.9, 17.3)
    Radio = st.sidebar.slider('Radio',37.8 , 39.3, 46.0)
    Newspaper = st.sidebar.slider('Newspaper', 69.2, 45.1, 70.1)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("iris_model.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
