import streamlit as st
import pickle

# Load the saved pipeline
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧠 Sentiment Analysis App")
user_input = st.text_input("Enter a sentence to analyze:")

if st.button("Predict"):
    if user_input.strip():
        prediction = model.predict([user_input])[0]
        sentiment = "😊 Positive" if prediction == 1 else "😠 Negative"
        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a valid sentence.")
