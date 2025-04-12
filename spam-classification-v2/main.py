from utils.data_loader import load_and_preprocess_data
from utils.preprocessor import prepare_features
from utils.visualization import plot_confusion_matrices, plot_accuracy_comparison
from models.decision_tree import create_model as create_dt_model
from models.naive_bayes import create_gaussian_model, create_multinomial_model
from models.logistic_regression import create_model as create_lr_model
from models.knn import create_model as create_knn_model
from sklearn.metrics import accuracy_score

def main():
    data = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = prepare_features(data)
    
    models = {
        'Decision Tree': create_dt_model(),
        'Gaussian NB': create_gaussian_model(),
        'Logistic Regression': create_lr_model(),
        'K-Nearest Neighbors': create_knn_model()
    }
    
    for name, model in models.items():
        if name == 'Gaussian NB':
            model.fit(X_train.toarray(), y_train)
        else:
            model.fit(X_train, y_train)
    
    plot_confusion_matrices(models, X_test, y_test)
    
    accuracies = {}
    for name, model in models.items():
        if name == 'Gaussian NB':
            predictions = model.predict(X_test.toarray())
        else:
            predictions = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, predictions)
    
    plot_accuracy_comparison(accuracies)
    
    for name, accuracy in accuracies.items():
        print(f"{name}: {accuracy:.4f}")

if __name__ == "__main__":
    main()