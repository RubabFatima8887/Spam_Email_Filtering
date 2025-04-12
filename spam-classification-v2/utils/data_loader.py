import pandas as pd
from google.colab import drive
from config import DATA_PATH

def load_and_preprocess_data():
    drive.mount('/content/drive')
    data = pd.read_csv(DATA_PATH)
    data.dropna(inplace=True)
    data['Body'] = data['Body'].str.strip()
    data['Body'] = data['Body'].str.replace('[^a-zA-Z0-9\s]', '')
    data['Label'] = pd.to_numeric(data['Label'])
    return data