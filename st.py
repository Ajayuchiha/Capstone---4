import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from transformers import pipeline, MarianMTModel, MarianTokenizer

# Load pre-trained components
MODEL = joblib.load('insurance_fraud_model.pkl')
SCALER = joblib.load('scaler.pkl')
IMPUTER = joblib.load('imputer.pkl')
LABEL_ENCODERS = joblib.load('label_encoders.pkl')
MODEL_FEATURES = joblib.load('model_features.pkl')

# NLP Models
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Translation Models
en_to_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_to_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_input_df(df, model_features):
    df = df.copy()
    for col in model_features:
        if col not in df.columns:
            df[col] = np.nan
    df = df[model_features]

    for col, le in LABEL_ENCODERS.items():
        if col in df.columns:
            known_classes = list(le.classes_)
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in known_classes else 0)

    df_imputed = IMPUTER.transform(df)
    df_scaled = SCALER.transform(df_imputed)
    return df_scaled

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Insurance AI Suite", layout="wide")
st.title("🏥 Insurance Fraud Detection, NLP & Translation")

claims_file = st.file_uploader("Upload Insurance Claims CSV", type=["csv"])
reviews_file = st.file_uploader("Upload Customer Reviews CSV", type=["csv"])

if claims_file:
    df_claims = pd.read_csv(claims_file)
    st.subheader("Fraud Prediction")
    if st.button("Predict Fraud Risk"):
        try:
            X = preprocess_input_df(df_claims, MODEL_FEATURES)
            predictions = MODEL.predict(X)
            df_claims['Predicted Fraud'] = predictions
            st.write(df_claims[['Predicted Fraud']].head())
            st.download_button("📥 Download Predictions", df_claims.to_csv(index=False), file_name="fraud_predictions.csv")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if reviews_file:
    df_reviews = pd.read_csv(reviews_file)
    st.subheader("📝 NLP: Sentiment & Summarization")

    def analyze_reviews(df):
        results = []
        for _, row in df.iterrows():
            text = str(row.get('ReviewText') or row.get('review_text') or '')[:512]
            if not text.strip():
                continue
            sentiment = sentiment_pipeline(text)[0]
            summary = summarizer(text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
            results.append({
                'original': text,
                'summary': summary,
                'sentiment_label': sentiment['label'],
                'sentiment_score': sentiment['score']
            })
        return pd.DataFrame(results)

    if st.button("Run NLP Analysis"):
        with st.spinner("Analyzing reviews..."):
            nlp_df = analyze_reviews(df_reviews)
            st.dataframe(nlp_df.head())
            st.download_button("📥 Download NLP Results", nlp_df.to_csv(index=False), file_name="nlp_reviews.csv")

    st.subheader("🌍 Translate Reviews to French")

    def translate_text(text):
        inputs = en_to_fr_tokenizer([text], return_tensors="pt", truncation=True, padding=True)
        translated = en_to_fr_model.generate(**inputs, max_length=512)
        return en_to_fr_tokenizer.decode(translated[0], skip_special_tokens=True)

    if st.button("Translate Reviews to French"):
        with st.spinner("Translating reviews to French..."):
            df_reviews['translated_text'] = df_reviews.iloc[:, 0].astype(str).apply(lambda x: translate_text(x[:512]))
            st.dataframe(df_reviews[['translated_text']].head())
            st.download_button("📥 Download Translations", df_reviews.to_csv(index=False), file_name="reviews_translated.csv")
