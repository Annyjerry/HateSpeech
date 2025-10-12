# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:27:08 2024

@author: Annyjerry
"""
import nltk
nltk.download('stopwords')

import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from joblib import load
import os

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'SVM.joblib')
file_path = os.path.join(script_dir, 'myDataset.xlsx')

# Load model and dataset
loaded_model = load(model_path)
df = pd.read_excel(file_path)

# Load stopwords
stop_words = set(stopwords.words('english'))

# Prepare TF-IDF vectorizer
df.dropna(subset=['Tweet'], inplace=True)
processed = df['Tweet'].str.lower()
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
tfidf_vectorizer = TfidfVectorizer(max_features=30000)
tfidf_vectorizer.fit(processed)

# --- Function to predict class ---
def predict_class(input_text):
    processed_text = input_text.lower()
    processed_text = ' '.join(term for term in processed_text.split() if term not in stop_words)
    text_features = tfidf_vectorizer.transform([processed_text])
    predicted_class = loaded_model.predict(text_features)
    class_mapping = {0: "Low", 1: "Medium", 2: "High"}
    return class_mapping[predicted_class[0]]

# --- Streamlit app ---
def main():
    st.title("🧠 Hate Speech Detection System")

    st.markdown("""
        Enter a statement below to check if it contains hate speech and its level of severity.
    """)

    input_text = st.text_area("Enter the text to classify:")

    if st.button("Predict"):
        if input_text.strip():
            predicted_class = predict_class(input_text)

            # Define background colors and messages
            if predicted_class == "Low":
                color = "#d4edda"  # light green
                text_color = "#155724"
                message = "✅ The word appears not be hate speech or low hate speech, but try to avoid using hate speech."
            elif predicted_class == "Medium":
                color = "#fff3cd"  # light yellow
                text_color = "#856404"
                message = "⚠️ This word is hate speech and is somewhat severe. Please avoid such words."
            else:  # High
                color = "#f8d7da"  # light red
                text_color = "#721c24"
                message = "🚫 This statement contains highly severe hate speech and has been flagged. You can get purnished for using such words."

            # Styled output box
            st.markdown(
                f"""
                <div style="background-color:{color}; color:{text_color}; padding:15px; border-radius:10px; margin-top:20px;">
                    <h3>Predicted Class: {predicted_class}</h3>
                    <p style="font-size:16px;">{message}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Please enter text for classification.")

if __name__ == "__main__":
    main()
