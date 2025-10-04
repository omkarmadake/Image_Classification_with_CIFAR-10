import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import cifar10

# CIFAR-10 class names
CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

def plot_samples(x_train, y_train, n=10):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.xticks([]); plt.yticks([])
        plt.imshow(x_train[i])
        plt.xlabel(CLASS_NAMES[y_train[i][0]])
    plt.show()

def plot_history(history, title="Training History"):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def count_predictions(y_true, y_pred):
    from collections import Counter
    true_counts = Counter(y_true.flatten())
    pred_counts = Counter(y_pred.flatten())

    print("\nTrue Image Counts per Class:")
    for i, c in enumerate(CLASS_NAMES):
        print(f"{c:10s}: {true_counts.get(i,0)}")

    print("\nPredicted Image Counts per Class:")
    for i, c in enumerate(CLASS_NAMES):
        print(f"{c:10s}: {pred_counts.get(i,0)}")
