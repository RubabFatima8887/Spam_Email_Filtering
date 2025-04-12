from sklearn.naive_bayes import MultinomialNB
from models.base_model import BaseModel

class NaiveBayesModel(BaseModel):
    def __init__(self):
        super().__init__(MultinomialNB())