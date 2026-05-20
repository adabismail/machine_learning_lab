import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


"""
make_blobs: Generates synthetic data for clustering and classification tasks.
n_samples: Total number of samples to generate.

centers: Number of centers to generate, or the fixed center locations.
To create a binary classification dataset, we set centers=2.
so that we have two distinct clusters corresponding to the two classes.
and we can seperate them with a linear decision boundary.

random_state: Controls the randomness of the generated data.
cluster_std: Standard deviation of the clusters.
A larger cluster_std value (e.g., 1.5) creates more spread-out clusters, which can make the classification task more challenging,
while a smaller value (e.g., 0.5) creates tighter clusters that are easier to classify.
"""
X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=1.5)


"""
train_test_split: Splits the dataset into training and testing sets.
X: The feature data.
y: The target labels.
test_size: Proportion of the dataset to include in the test split (e.g., 0.3 for 30% test data).
random_state: Controls the randomness of the split.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


"""
SGDClassifier: Stochastic Gradient Descent classifier, which can be used to train linear models.
loss: The loss function to be optimized. 'hinge' corresponds to the SVM loss function, which is used for linear SVMs.
random_state: Controls the randomness of the model training process.
fit: Fits the model to the training data (X_train and y_train).
"""
svm_model = SGDClassifier(loss='hinge', random_state=42)
svm_model.fit(X_train, y_train)



"""
coef_: The learned weight vector of the model, which indicates the importance of each feature in making predictions.
intercept_: The learned bias term of the model, which is added to the weighted sum of the features to make predictions.
"""
weights = svm_model.coef_[0]
bias = svm_model.intercept_[0]
print(f"Learned Weight Vector: {weights}")
print(f"Learned Bias: {bias}")
print("-" * 30)



"""
plot_svm_line: A function to visualize the decision boundary of the SVM model.
model: The trained SVM model.
X_data: The feature data used for plotting.
y_data: The target labels used for coloring the data points.
title: The title for the plot."""
def plot_svm_line(model, X_data, y_data, title):
    plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap='coolwarm', edgecolors='k')
    ax = plt.gca()
    xlim = ax.get_xlim()
    
    w = model.coef_[0]
    b = model.intercept_[0]
    x_points = np.linspace(xlim[0], xlim[1], 100)
    y_points = -(w[0] * x_points + b) / w[1]
    
    plt.plot(x_points, y_points, 'k-', linewidth=2)
    plt.title(title)
    plt.show()

# Plot the initial model
plot_svm_line(svm_model, X_train, y_train, "SVM Decision Boundary (Default Settings)")



C_values = [0.01, 0.1, 1, 10]


"""
C: Regularization parameter that controls the trade-off between maximizing the margin and minimizing classification errors.
A smaller C value (e.g., 0.01) allows for a wider margin but may lead to more misclassifications, while a larger C value (e.g., 10) focuses on minimizing misclassifications but may result in a narrower margin.
alpha: The regularization strength, which is the inverse of C (alpha = 1/C). A smaller alpha corresponds to a larger C, and vice versa.
"""
for C in C_values:
    alpha_value = 1.0 / C
    model_c = SGDClassifier(loss='hinge', alpha=alpha_value, random_state=42)
    model_c.fit(X_train, y_train)
    plot_svm_line(model_c, X_train, y_train, f"SVM Decision Boundary (C = {C})")


y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print("\n")


"""
confusion_matrix: Computes the confusion matrix to evaluate the performance of a classification model.
y_test: The true labels for the test set.
y_pred: The predicted labels from the model.
ConfusionMatrixDisplay: A utility to visualize the confusion matrix.
cmap: The colormap used for the display of the confusion matrix. 'Blues' is a common choice for better visual distinction between different values in the matrix.
"""
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()