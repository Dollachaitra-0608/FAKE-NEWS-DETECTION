# app.py
import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# UI
st.title("📰 Fake News Detector")
st.write("Enter the news content below and click the button to classify it.")

news_input = st.text_area("News Content")

if st.button("Predict"):
    if news_input:
        input_vec = vectorizer.transform([news_input])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.success("✅ This news is REAL.")
        else:
            st.error("🚨 This news is FAKE.")
    else:
        st.warning("Please enter some text.")
