import pandas as pd
from config import DATA_PATH

def load_data():
    """Load and preprocess the raw data"""
    data = pd.read_csv(DATA_PATH)
    data['Body'].fillna('', inplace=True)
    return data

def get_email_ids(data):
    """Extract email IDs from the data"""
    return data['Unnamed: 0']