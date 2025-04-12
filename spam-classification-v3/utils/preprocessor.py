from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import re

def load_and_preprocess():
    data = pd.read_csv(DATA_PATH)
    data = data.dropna()
    data['Message'] = data['Message'].apply(lambda x: re.sub('[^\w\s]', '', x.lower()))
    return data

def prepare_features(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Message'])
    y = data['Category']
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE), vectorizer