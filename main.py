
from utils.data_loader import load_data, get_email_ids
from utils.preprocessor import prepare_features
from utils.visualization import plot_emails_vs_labels

from models.linear_regression import LinearRegressionModel
from models.naive_bayes import NaiveBayesModel
from models.knn import KNNModel
from models.decision_tree import DecisionTreeModel

def main():
    data = load_data()
    X_train, X_test, y_train, y_test = prepare_features(data)
    emails = get_email_ids(data)
    
    plot_emails_vs_labels(emails, data['Label'])
    
    models = {
        "Linear Regression": LinearRegressionModel(),
        "Naive Bayes": NaiveBayesModel(),
        "K-Nearest Neighbors": KNNModel(),
        "Decision Tree": DecisionTreeModel()
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.train(X_train, y_train)
        accuracy = model.evaluate(X_test, y_test)
        results[name] = accuracy
    
    print("\n=== Model Comparison ===")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")

if __name__ == "__main__":
    main()