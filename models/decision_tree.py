from sklearn.tree import DecisionTreeClassifier
from models.base_model import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__(DecisionTreeClassifier())