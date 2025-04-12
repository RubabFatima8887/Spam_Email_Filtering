import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from config import DATA_PATH, SPECIAL_CHARS

def load_data():
    data = pd.read_csv(DATA_PATH)[['Category', 'Message']]
    data.drop_duplicates(inplace=True)
    data['Message'].fillna("", inplace=True)
    return data

def extract_features(data):
    data['Text_Length'] = data['Message'].apply(len)
    data['Word_Count'] = data['Message'].apply(lambda x: len(x.split()))
    data['Special_Char_Count'] = data['Message'].apply(lambda x: sum(x.count(char) for char in SPECIAL_CHARS))
    data['Capitalized_Word_Count'] = data['Message'].apply(lambda x: sum(1 for word in x.split() if word[0].isupper()))
    return data

def prepare_features(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Message'])
    y = LabelEncoder().fit_transform(data['Category'])
    return X, y, vectorizer