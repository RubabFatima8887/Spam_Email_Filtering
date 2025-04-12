import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def prepare_data(data):
    data = data.dropna().fillna(data.median())
    preprocessed_emails = [preprocess_text(email) for email in data['Body']]
    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(preprocessed_emails)
    return features, data['Label'], tfidf