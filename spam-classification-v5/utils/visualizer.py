import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(metrics):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(metrics.keys(), [m['accuracy'] for m in metrics.values()])
    plt.title('Model Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.bar(metrics.keys(), [m['time'] for m in metrics.values()])
    plt.title('Training Time (s)')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrices(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for (name, metric), ax in zip(metrics.items(), axes):
        sns.heatmap(metric['cm'], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(name)
    plt.tight_layout()
    plt.show()