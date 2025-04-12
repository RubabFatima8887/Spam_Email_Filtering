from google.colab import drive
from utils.preprocessor import load_and_preprocess, prepare_features
from utils.evaluator import evaluate_model
from models.naive_bayes import train_model as train_nb
from models.decision_tree import train_model as train_dt
from models.knn import train_model as train_knn
from config import SAMPLE_EMAILS

def main():
    drive.mount('/content/drive')
    data = load_and_preprocess()
    (X_train, X_test, y_train, y_test), vectorizer = prepare_features(data)
    
    nb_model = train_nb(X_train, y_train)
    dt_model = train_dt(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    
    for name, model in [('Naive Bayes', nb_model), 
                       ('Decision Tree', dt_model),
                       ('KNN', knn_model)]:
        evaluate_model(model, X_test, y_test, name)
    
    samples = vectorizer.transform(SAMPLE_EMAILS)
    print("\nSample Predictions:")
    for email, nb_pred, dt_pred, knn_pred in zip(
        SAMPLE_EMAILS, 
        nb_model.predict(samples),
        dt_model.predict(samples),
        knn_model.predict(samples)
    ):
        print(f"\nEmail: {email}")
        print(f"NB: {nb_pred} | DT: {dt_pred} | KNN: {knn_pred}")

if __name__ == "__main__":
    main()