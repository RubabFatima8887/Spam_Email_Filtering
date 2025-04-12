from google.colab import drive
import pandas as pd
from utils.preprocessor import prepare_data
from models import knn, naive_bayes, decision_tree
from utils.evaluator import evaluate
from utils.visualizer import plot_metrics, plot_confusion_matrices
from config import DATA_PATH, TEST_SIZE, RANDOM_STATE, SAMPLE_EMAIL

def main():
    drive.mount('/content/drive')
    data = pd.read_csv(DATA_PATH)
    
    X, y, vectorizer = prepare_data(data)
    
    models = {
        'KNN': knn.train_model(X, y),
        'Naive Bayes': naive_bayes.train_model(X, y),
        'Decision Tree': decision_tree.train_model(X, y)
    }
    
    metrics = {name: evaluate(model, X, y) for name, model in models.items()}
    
    for name, metric in metrics.items():
        print(f"{name}: Accuracy={metric['accuracy']:.4f}, Time={metric['time']:.2f}s")
    
    plot_metrics(metrics)
    plot_confusion_matrices(metrics)
    
    sample_features = vectorizer.transform([SAMPLE_EMAIL])
    prediction = models['Decision Tree'].predict(sample_features)[0]
    print("\nSample Email Prediction:", "SPAM" if prediction else "NOT SPAM")

if __name__ == "__main__":
    main()