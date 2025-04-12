import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_metrics(results):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(results.keys(), [r['accuracy'] for r in results.values()])
    plt.title('Model Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.bar(results.keys(), [r['time'] for r in results.values()])
    plt.title('Training Time (s)')
    
    plt.tight_layout()
    plt.show()