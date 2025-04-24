import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create the vectorizer (if not saved with model)
vectorizer = CountVectorizer()

# Function to predict sentiment
def predict_sentiment(text):
    # Transform input text into feature vector
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return 'Positive' if prediction == 1 else 'Negative'

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter text to analyze its sentiment:")

# User input
user_input = st.text_area("Input Text:")

if st.button("Analyze Sentiment"):
    if user_input:
        result = predict_sentiment(user_input)
        st.write(f"Sentiment: {result}")
    else:
        st.write("Please enter some text.")
