from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name):
    pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    cm = metrics.confusion_matrix(y_test, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()