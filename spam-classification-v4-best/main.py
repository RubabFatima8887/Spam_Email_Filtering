from google.colab import drive
from utils.preprocessor import load_data, extract_features, prepare_features
from models import decision_tree, naive_bayes, knn
from utils.evaluator import evaluate
from utils.visualizer import plot_confusion_matrix, plot_metrics
from config import TEST_EMAILS

def main():
    drive.mount('/content/drive')
    
    data = load_data()
    data = extract_features(data)
    X, y, vectorizer = prepare_features(data)
    
    models = {
        'Decision Tree': decision_tree.train(X, y),
        'Naive Bayes': naive_bayes.train(X, y),
        'KNN': knn.train(X, y)
    }
    
    tree_info = decision_tree.inspect_tree(models['Decision Tree'])
    print(f"Root Feature: {tree_info['root_feature']}")
    print(f"Root Threshold: {tree_info['root_threshold']}")
    
    results = {name: evaluate(model, X, y) for name, model in models.items()}
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Time: {result['time']:.4f}s")
        plot_confusion_matrix(result['cm'], name)
    
    plot_metrics(results)
    
    test_vectors = vectorizer.transform(TEST_EMAILS)
    for name, model in models.items():
        print(f"\n{name} Predictions:")
        print(model.predict(test_vectors))

if __name__ == "__main__":
    main()