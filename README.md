# GoEmotions Emotion Detector (BiLSTM)

▶️ **Live App:** https://goemotions-nw3wxduxmxib7xxtcbm6te.streamlit.app/

This project is a **multi-label emotion classification system** trained on the
Google Research **GoEmotions dataset**, capable of detecting **28 different emotions**
from text, including:

Joy, Love, Anger, Fear, Gratitude, Surprise, Neutral, and more.

---

## 🚀 Features

✅ 28-emotion multi-label classifier  
✅ BiLSTM deep learning model  
✅ Custom tokenizer and vocabulary  
✅ Multi-hot label encoding  
✅ BCEWithLogitsLoss training  
✅ Evaluation metrics:
- Micro F1
- Macro F1
- Subset Accuracy

✅ Real-time Streamlit Web App  
✅ Probability visualization  
✅ Ranked emotion output (top + 5)

---

## 🧠 Model Performance

Validation Results:

MICRO F1 : 0.4708  
MACRO F1 : 0.3786  
SUBSET ACCURACY : 0.3287  
Training Accuracy : ~76%

These scores are considered strong for a BiLSTM model on the GoEmotions dataset.

---

## 🏗️ Tech Stack

- Python
- PyTorch
- Streamlit
- Plotly
- NLTK
- NumPy

---

## 🖥️ Web App

The app includes:

- Top emotion prediction
- Probability bars
- Emotion distribution chart
- Sample text inputs
- Model capability overview

---

## 🔧 Installation

pip install -r requirements.txt

Download or place the checkpoint file:

goemotions_bilstm_checkpoint.pth

---

## ▶️ Run the App

streamlit run app.py

---

## 📦 Model Deployment

The model is saved as:

goemotions_bilstm_checkpoint.pth

containing:

- model weights
- vocabulary
- max sequence length
- threshold

Loaded once through:

from emotions_backend import find_emotions

---

## ✅ Status

This is a **portfolio-ready NLP project** demonstrating:

- deep learning
- multi-label classification
- evaluation metrics
- deployment
- app development

---
