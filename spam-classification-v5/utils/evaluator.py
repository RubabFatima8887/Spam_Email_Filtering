import time
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(model, X_test, y_test):
    start = time.time()
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'time': time.time() - start,
        'cm': confusion_matrix(y_test, y_pred)
    }