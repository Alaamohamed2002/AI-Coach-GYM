import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, title):
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()

def error_analysis(y_true, y_pred):
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")

def result_analysis(y_true, y_pred):
   
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    print("Confusion Matrix (Percentage):")
    print(cm_percentage)
    
    print("\nDetailed Analysis:")
    for i in range(len(cm)):
        print(f"Class {i}:")
        print(f"  True Positives (TP): {cm[i, i]}")
        print(f"  False Positives (FP): {cm[:, i].sum() - cm[i, i]}")
        print(f"  False Negatives (FN): {cm[i, :].sum() - cm[i, i]}")
        print(f"  True Negatives (TN): {cm.sum() - (cm[:, i].sum() + cm[i, :].sum() - cm[i, i])}")
        print(f"  Precision: {cm[i, i] / (cm[:, i].sum()):.2f}")
        print(f"  Recall: {cm[i, i] / (cm[i, :].sum()):.2f}")
        print(f"  F1 Score: {2 * cm[i, i] / (cm[i, :].sum() + cm[:, i].sum()):.2f}")

def visualize_results(X, y_true, y_pred, title):
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_true, style=y_pred, palette='Set1', s=100)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(title='Classes')
    plt.show()

def main():
    
    iris = load_iris()
    X = iris.data
    y = iris.target

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

   
    y_train_pred = model.predict(X_train)
    
    y_test_pred = model.predict(X_test)

    
    plot_confusion_matrix(y_train, y_train_pred, "Confusion Matrix for Training Set")
    
    plot_confusion_matrix(y_test, y_test_pred, "Confusion Matrix for Test Set")

   
    print("Error Analysis for Training Set")
    error_analysis(y_train, y_train_pred)
    
    
    print("\nError Analysis for Test Set")
    error_analysis(y_test, y_test_pred)

    
    print("\nResult Analysis for Training Set")
    result_analysis(y_train, y_train_pred)
    
    
    print("\nResult Analysis for Test Set")
    result_analysis(y_test, y_test_pred)

   
    visualize_results(X_train, y_train, y_train_pred, "Training Set Results")
    
    visualize_results(X_test, y_test, y_test_pred, "Test Set Results")

if __name__ == "__main__":
    main()
