import pandas as pd
import streamlit as st
import numpy as np
import pickle


#Load the model
model = pickle.load(open("model.pkl", "rb"))

st.title("Iris Flower Classifier")

#Input fields

sepal_length = st.text_input("Sepal Length")
sepal_width = st.text_input("Sepal width")
petal_length = st.text_input("Petal Length")
petal_width = st.text_input("Petal Width")


#Function to make prediction
def predict(sepol_length, sepal_width, petal_length, petal_width):
    try:
        # convert inputs to float and create DataFrame
        input_data = {
            "sepal_length" : float(sepal_length),
            "sepal_width" : float(sepal_width),
            "petal_length" : float(petal_length),
            "petal_width" : float(petal_width)
        }

        X = pd.DataFrame([input_data])
        #Make prediction
        prediction = model.predict(X)[0]

        #Display result
        flower_types = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        flower_name = flower_types.get(prediction, "Unknown")
        st.success(f"This Iris Flower SPecies is {flower_name}")
    except ValueError:
        st.error("Please enter valid numbers")

# Button to make prediction
if st.button("Predict"):
    predict(sepal_length, sepal_width, petal_length, petal_width)
