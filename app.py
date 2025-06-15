from flask import Flask, render_template, request
import joblib
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Initialize NLP components
nltk.download(['stopwords', 'punkt', 'wordnet'])
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# AI-specific terms
AI_TERMS = {
    'ai', 'artificial intelligence', 'machine learning', 'deep learning',
    'neural network', 'chatbot', 'llm', 'gpt', 'openai', 'ethics',
    'bias', 'automation', 'job displacement', 'regulation', 'privacy'
}

# Load models
try:
    model = joblib.load('ai_sentiment_model.joblib')
    vectorizer = joblib.load('ai_text_vectorizer.joblib')
except FileNotFoundError:
    model = None
    vectorizer = None
    print("AI model files not found.")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    tokens = [w for w in tokens if w in AI_TERMS or len(w) > 2]
    return " ".join(tokens)

@app.route('/', methods=['GET', 'POST'])
def home():
    sentiment = None
    confidence = None
    ai_topics = []
    result_class = ""

    if request.method == 'POST':
        text = request.form.get('text', '').strip()

        if not text:
            sentiment = "Please enter text about AI to analyze"
            result_class = "info-message"
        elif model and vectorizer:
            processed_text = preprocess_text(text)
            
            # Extract AI topics mentioned
            ai_topics = [term for term in AI_TERMS if term in processed_text]
            
            if not ai_topics:
                sentiment = "No clear AI topics detected. Please mention specific AI technologies or issues."
                result_class = "info-message"
            else:
                vectorized_text = vectorizer.transform([processed_text])
                prediction = model.predict(vectorized_text)[0]
                
                sentiment_map = {
                    3: "Very Positive (Enthusiastic)",
                    2: "Positive",
                    1: "Neutral",
                    0: "Concerned",
                    -1: "Negative",
                    -2: "Very Negative (Skeptical)"
                }
                
                sentiment_label = sentiment_map.get(prediction, "Unknown")
                probabilities = model.predict_proba(vectorized_text)[0]
                confidence = max(probabilities) * 100
                
                sentiment = f"Sentiment: {sentiment_label}"
                if confidence:
                    sentiment += f" | Confidence: {confidence:.1f}%"
                if ai_topics:
                    sentiment += f" | Topics: {', '.join(ai_topics)}"
                
                result_class = "result-" + sentiment_label.split()[0].lower()

    return render_template('index.html', 
                         sentiment=sentiment, 
                         confidence=confidence,
                         ai_topics=ai_topics,
                         result_class=result_class)

if __name__ == '__main__':
    app.run(debug=True)