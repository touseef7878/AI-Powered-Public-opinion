import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np

# Download NLTK data
nltk.download(['stopwords', 'punkt', 'wordnet'])

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# AI-specific keywords and terms
AI_KEYWORDS = {
    'ai', 'artificial intelligence', 'machine learning', 'deep learning', 
    'neural network', 'chatbot', 'llm', 'gpt', 'openai', 'ethics',
    'bias', 'automation', 'job displacement', 'regulation', 'privacy'
}

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')]
    tokens = [w for w in tokens if w in AI_KEYWORDS or len(w) > 2]
    return " ".join(tokens)

# Load dataset
try:
    data = pd.read_csv('ai_opinion_data.csv')
    data = data[['sentiment', 'text', 'ai_topic']].dropna()
    
    # Enhanced sentiment mapping for AI opinions
    sentiment_mapping = {
        'enthusiastic': 3,
        'positive': 2,
        'neutral': 1,
        'concerned': 0,
        'negative': -1,
        'skeptical': -2
    }
    data['sentiment'] = data['sentiment'].map(sentiment_mapping)
    
except FileNotFoundError:
    print("Error: ai_opinion_data.csv not found. Please create the file first.")
    exit()

# Preprocess text
data['processed_text'] = data['text'].apply(preprocess_text)

# Split data
X = data['processed_text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Evaluate
y_pred = model.predict(X_test_vectorized)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Get only the classes that actually appear in the test data
present_classes = sorted(set(y_test))
present_class_names = [k for k, v in sentiment_mapping.items() if v in present_classes]

# 1. Detailed Classification Report
print("\n=== Classification Report ===")
print(classification_report(
    y_test, 
    y_pred,
    target_names=present_class_names,
    labels=present_classes,
    zero_division=0
))

# Cross-Validation for Robustness
cv_scores = cross_val_score(model, X_train_vectorized, y_train, cv=5, scoring='accuracy')
print(f"\n=== Cross-Validation Scores ===")
print(f"Mean Accuracy: {cv_scores.mean():.2f} (±{cv_scores.std():.2f})")

# Save model
joblib.dump(model, 'ai_sentiment_model.joblib')
joblib.dump(vectorizer, 'ai_text_vectorizer.joblib')
print("\n✅ AI Sentiment Model saved successfully!")