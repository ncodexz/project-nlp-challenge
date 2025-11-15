import re
import nltk
import joblib
from nltk.corpus import stopwords

# Download stopwords if not available
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load model and vectorizer
try:
    model = joblib.load('/Users/nazb/VSCode101/project-nlp-challenge-1/Deployment/logistic_regression_model.pkl')
    vectorizer = joblib.load('/Users/nazb/VSCode101/project-nlp-challenge-1/Deployment/tfidf_vectorizer.pkl')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    vectorizer = None

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    
    # ADDED: Handle acronyms with dots
    acronyms_pattern = r'\b(?:[a-z]\.){2,}'
    found_acronyms = re.findall(acronyms_pattern, text)
    for abbr in found_acronyms:
        replacement = abbr.replace('.', '')
        text = text.replace(abbr, replacement)
    
    # Your existing cleaning code
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

def predict_news(title, text):
    """Main prediction function"""
    if model is None or vectorizer is None:
        return "Error: Models not loaded", "0%"
    
    try:
        # Combine title and text as in training
        content = f"{title} {text}"
        
        # Clean text
        cleaned_text = clean_text(content)
        
        # Vectorize
        text_vector = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        # Get confidence
        confidence = probability[prediction] * 100
        
        # Determine label
        label = "REAL" if prediction == 1 else "FAKE"
        
        return label, f"{confidence:.1f}%"
    
    except Exception as e:
        return f"Error: {str(e)}", "0%"