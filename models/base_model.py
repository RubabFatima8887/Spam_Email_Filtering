# models/base_model.py
from sklearn.metrics import accuracy_score
from utils.visualization import plot_confusion_matrix, plot_roc_curve

class BaseModel:
    def __init__(self, model):
        self.model = model
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Visualization
        plot_confusion_matrix(y_test, y_pred)
        plot_roc_curve(y_test, y_pred)
        
        return accuracy