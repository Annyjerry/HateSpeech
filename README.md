# 🧠 Hate Speech Detection System

This is a Streamlit web application for detecting and classifying hate speech from user input text using a Support Vector Machine (SVM) model.

It categorizes text into three levels of severity — Low, Medium, and High — with color-coded warnings and explanations.



## 🚀 Features

- 🟢 Low: Indicates mild or non-hate speech.  
  _"The word appears not to be hate speech, but try to avoid using hate-related terms."_

- 🟡 Medium: Indicates moderate hate speech severity.  
  _"This word is hate speech and somewhat severe. Please avoid such words."_

- 🔴 High: Indicates strongly offensive or harmful hate speech.  
  _"This statement contains highly severe hate speech and has been flagged. You can get punished for using such words."_



## 🧩 Technologies Used

- Python 3.11+
- Streamlit — for building the web interface  
- NLTK — for stopword removal and text preprocessing  
- Pandas — for data handling and loading dataset  
- Scikit-learn (SVM) — for the machine learning model  
- Joblib — for saving and loading the trained model  
- OpenPyXL — for reading Excel datasets

⚙️ Installation and Setup

1️⃣ Clone the repository
bash
git clone https://github.com/Annyjerry/HateSpeech.git
cd hate-speech-detection

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run Cyber.py

👨‍💻 Author
Name: Anny Jerry
Created: January 2024
Updated: October 2025

🛡️ License

This project is licensed under the MIT License — you can freely use and modify it for educational and research purposes.

