import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from imblearn.over_sampling import SMOTE
from transformers import pipeline, MarianMTModel, MarianTokenizer

# -------------------------------
# Load Data
# -------------------------------

df_structured = pd.read_csv(r"C:\Users\AJAY\Downloads\PROJECT\capst_4\Data\insurance_claims.csv")
df_reviews = pd.read_csv(r"C:\Users\AJAY\Downloads\PROJECT\capst_4\Data\insurance_customer_reviews_gemini_enhanced.csv")

# -------------------------------
# PART 1: ML – Risk/Fraud Classification
# -------------------------------

def preprocess_structured_data(df, target_column='fraud_reported'):
    df = df.copy()
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' not found in dataset columns.")

    # Drop obvious ID-like columns
    id_cols = [col for col in df.columns if "id" in col.lower()]
    df.drop(columns=id_cols, errors='ignore', inplace=True)

    # Convert object columns to numeric using label encoding
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            df[col] = df[col].astype(str)  # Keep target as string for now

    return df, label_encoders

def train_model(df, target_column='fraud_reported'):
    df, label_encoders = preprocess_structured_data(df, target_column=target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode target
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
    print("✅ Classification Report:\n", classification_report(y_test, y_pred))

    # Save trained components
    joblib.dump(model, 'insurance_fraud_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    expected_features = X.columns.tolist()
    joblib.dump(expected_features, 'expected_features.pkl')
    joblib.dump(list(X.columns), "model_features.pkl")
    
    return model

# Run training
print("\n🔍 Training model on structured insurance data...\n")
model = train_model(df_structured)

# -------------------------------
# PART 2: NLP – Sentiment + Summarization
# -------------------------------

def analyze_reviews(df_reviews):
    df = df_reviews.copy()

    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    results = []
    for _, row in df.iterrows():
        text = str(row.get('ReviewText', ''))[:512]

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

print("\n📝 Running sentiment analysis and summarization on reviews...\n")
review_results = analyze_reviews(df_reviews)
review_results.to_csv("review_nlp_results.csv", index=False)
print("✅ NLP Results saved to 'review_nlp_results.csv'")

# -------------------------------
# PART 3: Translation (English <-> French)
# -------------------------------

# Load translation models
en_to_fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
en_to_fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

fr_to_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
fr_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

def translate_text(text, tokenizer, model):
    inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs, max_length=512)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_reviews(df, source_lang='en'):
    df = df.copy()
    translations = []

    for text in df['review_text']:
        text = str(text)[:512]

        if source_lang == 'en':
            translated = translate_text(text, en_to_fr_tokenizer, en_to_fr_model)
        else:
            translated = translate_text(text, fr_to_en_tokenizer, fr_to_en_model)

        translations.append(translated)

    df['translated_text'] = translations
    return df

print("\n🌍 Translating reviews from English to French...\n")
df_reviews_fr = translate_reviews(df_reviews, source_lang='en')
df_reviews_fr.to_csv("reviews_translated_en_to_fr.csv", index=False)
print("✅ Translated reviews (EN→FR) saved to 'reviews_translated_en_to_fr.csv'")
