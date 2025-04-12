from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def prepare_features(data):
    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(data['Body'])
    labels = data['Label']
    return train_test_split(features, labels, test_size=0.2, random_state=42)