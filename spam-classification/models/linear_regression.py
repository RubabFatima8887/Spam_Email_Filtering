from sklearn.linear_model import LinearRegression
from models.base_model import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(LinearRegression())
    
    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return [1 if pred >= 0.5 else 0 for pred in y_pred]