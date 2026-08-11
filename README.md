#  Hate Speech Detection System

A Streamlit web app that classifies user input text into **Low**, **Medium**, or **High** severity levels of hate speech, using a trained SVM model with TF-IDF feature extraction.

🔗 Live demo: https://hatespeech-classifier.streamlit.app

⚠️ **Disclaimer**
This is an educational/research project. The model was trained on a limited dataset and is not suitable for real-world content moderation without further validation. It may reflect biases present in the training data.

## Features
- 🟢 Low: mild or non-hate speech, flagged with a gentle caution
- 🟡 Medium: moderate severity, flagged with a warning
- 🔴 High: strongly offensive or harmful, flagged clearly

## Tech Stack
- Python 3.11+
- Streamlit — web interface
- NLTK — text preprocessing (stopword removal, tokenization)
- Scikit-learn (SVM) — classification model
- TF-IDF — feature extraction
- Joblib — model persistence
- OpenPyXL — reading the training dataset

## Installation
```bash
git clone https://github.com/Annyjerry/HateSpeech.git
cd HateSpeech

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run Cyber.py
```

## Author
Aniebiet Jeremiah
Created: January 2024 · Updated: October 2025

## License
MIT License — free to use and modify for educational and research purposes.