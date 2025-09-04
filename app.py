import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("Phishing URL Detector")
st.write("Enter a URL below to check if it's legitimate or a phishing attempt.")

# Input from user
url_input = st.text_input("Enter URL here", "")

if url_input:
    # Vectorize input
    input_vector = vectorizer.transform([url_input])
    prediction = model.predict(input_vector)[0]

    if prediction == 1:
        st.error("Phishing detected!")
    else:
        st.success("This URL looks safe.")
