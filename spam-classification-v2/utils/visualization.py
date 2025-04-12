import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for (name, model), ax in zip(models.items(), axes.flat):
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(name)
    plt.tight_layout()
    plt.show()

def plot_accuracy_comparison(accuracies):
    plt.bar(accuracies.keys(), accuracies.values())
    plt.xlabel('Classification Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Classification Models')
    plt.show()