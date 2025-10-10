# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:27:08 2024

@author: Annyjerry
"""

import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from joblib import load

# Load the pre-trained SVM model
import os

# Get the absolute path to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'SVM.joblib')

# Load the SVM model
loaded_model = load(model_path)

#loaded_model = load('SVM.joblib')

import os

# Get the absolute path to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'myDataset.xlsx')

# Read Excel file
df = pd.read_excel(file_path)

# Function to preprocess input text and predict class
def predict_class(input_text):
    processed_text = input_text.lower()
    processed_text = ' '.join(term for term in processed_text.split() if term not in stop_words)
    text_features = tfidf_vectorizer.transform([processed_text])
    predicted_class = loaded_model.predict(text_features)
    class_mapping = {0: "Low", 1: "Midium", 2: "High"}
    return class_mapping[predicted_class[0]]

# Streamlit web application
def main():
    st.title("Text Classification Demo")

    # Get user input
    input_text = st.text_area("Enter the text to classify:")

    if st.button("Predict"):
        if input_text:
            # Preprocess and predict
            predicted_class = predict_class(input_text)
            st.success(f"The predicted class for the input text is: {predicted_class}")
        else:
            st.warning("Please enter text for classification.")

if __name__ == "__main__":
    # Load stopwords and TF-IDF vectorizer
    stop_words = set(stopwords.words('english'))
    df = pd.read_excel('myDataset.xlsx')
    df.dropna(subset=['Tweet'], inplace=True)
    processed = df['Tweet'].str.lower()
    processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
    tfidf_vectorizer = TfidfVectorizer(max_features=30000)
    tfidf_vectorizer.fit(processed)

    main()
