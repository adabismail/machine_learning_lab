from sklearn.datasets import fetch_california_housing # skit-learn constains numerous datasets for testing and learning purposes.
import numpy as np # Numerical computing.
import matplotlib # Creating Data visualizations.
matplotlib.use('TkAgg') # Use TkAgg backend for matplotlib to enable plotting in environments without a display server.
import matplotlib.pyplot as plt # Creating Data visualizations.
import pandas as pd # Data handling & manipulation.

# dataset is loaded using inbuild fetch_california_housing() function from sklearn.datasets.
# This dataset contains information about California housing prices and is commonly used for regression tasks in machine learning.
dataset = fetch_california_housing()

"""
    Dataframe is a 2D data structure in pandas that can hold data of different types (like integers, floats, strings, etc.) in columns. 
    It is similar to a table in a relational database or an Excel spreadsheet.
    Each column in a DataFrame can be thought of as a Series, and the entire DataFrame can be thought of as a collection of Series objects. 
    The 'columns' parameter is used to specify the names of the columns in the DataFrame
 """
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

""" 
    Function to normalize the data using mean and standard deviation.
      -----------   Why Normalization?   ----------------
    Take Example :   Number of Bedrooms: Usually a small number (1, 2, 3, 4).Square Footage: A much larger number (800, 1500, 3000,2000).
    Because 3000 is mathematically much larger than 3, the model might mistakenly think the square footage is 1000 times more important than the number of bedrooms.
    ----------   How Normalization Fixes This ------
    To normalize, we squeeze both sets of numbers into small numbers usually between and around -1 and 1.
    Take the above two examples as input features and we will apply normalization to them.
    mean_bedrooms = 2.5, std_bedrooms = 1.0, mean_square_footage = 1824, std_square_footage = 1000.
    Normalized Bedroom  = (3 - 2.5) / 1.0 = 0.5, Normalized Square Footage = (3000 - 1500) / 1000 = 1.5.
    Feature          Original                ValuesStandardized (normalized)
    Bedrooms        [1, 2, 3, 4]               [-1.34, -0.45, 0.45, 1.34]
    Sq Footage      [800, 1500, 3000, 2000]    [-1.28, -0.41, 1.47, 0.22]
 """

def normalize(X):
    X_arr = np.array(X)
    if len(X_arr.shape) == 1:
        X_arr = X_arr.reshape(-1, 1)
    mean = np.mean(X_arr, axis=0)
    std = np.std(X_arr, axis=0)
    std_replaced = np.where(std == 0, 1, std)
    return (X_arr - mean) / std_replaced, mean, std

"""
    Model1 : As the problem asked to train the model1 on one input feature only, we decided to take 'MedInd' as the one..
    The normalize function will give X_one means the input feature vector,mean_one as the mean of that feature
    and std_one as the standard deviation of that feature. We will use these mean and std values later to denormalize the predictions if needed.
    Model2 : For the second model, we will use all the features in the dataset.
    The normalize function will give X_all as the input feature matrix, mean_all as the mean of each feature and std_all as the standard deviation of each feature.
    The target variable (house prices) is stored in y, which is reshaped to be a column vector for compatibility with our linear regression implementation.
"""

X_one, mean_one, std_one = normalize(df[['MedInc']].values)
X_all, mean_all, std_all = normalize(df.values)
y = dataset.target.reshape(-1, 1)

#  Splitting the dataset into training and testing sets (80 / 20)
np.random.seed(42)
idx = np.random.permutation(len(df))
split = int(0.8 * len(df))
tr, te = idx[:split], idx[split:]

X1_train, X1_test = X_one[tr], X_one[te]
X2_train, X2_test = X_all[tr], X_all[te]
y_train, y_test = y[tr], y[te]

"""
     Class definition for linear regression model.
     The linearRegression class implements a simple linear regression model using gradient descent for optimization.
     The class has the following methods:
     - __init__: Initializes the model with specified learning rate and number of iterations, 
     and sets initial weights, bias, and history lists for loss and parameters.
 """

class linearRegression:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.lr = learning_rate
        self.iters = iterations
        self.weights = None
        self.bias = 0
        self.loss_history = []
        self.theta_history = []

    """
    The fit method trains the linear regression model on the provided input features (X) and target variable (y).
    It uses gradient descent to optimize the weights and bias of the model.
    The method performs the following steps:
1. Reshapes the input features if they are one-dimensional.
2. Initializes the weights and bias to zero.
3. Iteratively updates the weights and bias for a specified number of iterations:
   a. Computes the predicted values (Y_pred) using the current weights and bias.
   b. Calculates the error between the predicted values and the actual target values.
   c. Computes the mean squared error loss and appends it to the loss history.
   d. If the model has only one feature, it appends the current bias and weight to the theta history for visualization purposes.
   e. Computes the gradients for weights (dw) and bias (db) and updates them using the learning rate.
"""

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        self.loss_history = []
        self.theta_history = []

        for i in range(self.iters):
            Y_pred = np.dot(X, self.weights) + self.bias
            
            error = Y_pred - y
            loss = (1 / (2 * n_samples)) * np.sum(error**2)
            self.loss_history.append(loss)
            
            if n_features == 1:
                # Ensure we extract standard float values for history
                self.theta_history.append((float(self.bias), float(self.weights[0, 0])))

            # Compute gradients and move towards better decision boundary..
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return np.dot(X, self.weights) + self.bias


    #  Added mse and r2 methods missing  to evaluate the models
    def mse(self, X, y):
        return float(np.mean((self.predict(X) - y) ** 2))

    def r2(self, X, y):
        ss_res = np.sum((self.predict(X) - y) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

"""
     We create two instances of the linearRegression class, m1 and m2, for the two models.
     We call the fit method on both models to train them using the respective input features (X_one for model 1 and X_all for model 2) and the target variable (y).
     """
#  Train on the training splits rather than the entire dataset
m1 = linearRegression(learning_rate=0.01, iterations=1000)
m1.fit(X1_train, y_train)

m2 = linearRegression(learning_rate=0.01, iterations=1000)
m2.fit(X2_train, y_train)

print("\n===== Model Comparison =====")
print(f"Model 1 (1 feature)  — Test MSE: {m1.mse(X1_test, y_test):.4f}  |  Test R²: {m1.r2(X1_test, y_test):.4f}")
print(f"Model 2 (8 features) — Test MSE: {m2.mse(X2_test, y_test):.4f}  |  Test R²: {m2.r2(X2_test, y_test):.4f}")
print("=> Model 2 has more features so it captures more variance → lower MSE, higher R².")

"""
     We create several plots to visualize the results of our linear regression models:
1. Loss vs Iterations: This plot shows how the mean squared error loss decreases over iterations for both models. It helps us understand the convergence of the training process.
2. Model 1: MedInc vs Price: This scatter plot shows the relationship between the normalized median income (MedInc) and the house prices (y) for the first 200 samples. The red line represents the regression line predicted by model 1.
3. Model 1: Loss Contour and GD Path: This contour plot shows the loss landscape for model 1 with respect to its bias and weight parameters. The red path indicates the trajectory of the parameters during gradient descent optimization.
4. Effect of Learning Rates on Convergence: This plot compares the loss curves for different learning rates, showing how the choice of learning rate affects the convergence of the training process.
     Each plot is saved as a PNG file for later reference.
     """

plt.figure(figsize=(10, 5))
plt.plot(m1.loss_history, label=f'Model 1 (1 Feature) Final: {m1.loss_history[-1]:.4f}')
plt.plot(m2.loss_history, label=f'Model 2 (8 Features) Final: {m2.loss_history[-1]:.4f}')
plt.title("Loss vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig('loss_comparison.png')

plt.figure(figsize=(8, 5))



#  Plot the testing data rather than the raw X_one sequence to properly evaluate
sample = np.argsort(X1_test[:, 0])[:200]
plt.scatter(X1_test[sample, 0], y_test[sample], alpha=0.5, label='Actual Data')
plt.plot(X1_test[sample, 0], m1.predict(X1_test[sample]), color='red', linewidth=3, label='Regression Line')
plt.title("Model 1: MedInc vs Price")
plt.xlabel("Normalized Median Income")
plt.ylabel("House Price")
plt.legend()
plt.savefig('plot2.png')




#  Added Plot 3  (Predicted vs Actual for Model 2)
plt.figure(figsize=(8, 5))
y2_pred = m2.predict(X2_test).flatten()
plt.scatter(y_test.flatten(), y2_pred, alpha=0.3, s=15, color='tomato', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', linewidth=1.5, label='Perfect fit')
plt.title("Model 2: Predicted vs Actual (test set)")
plt.xlabel("Actual Price ($100k)")
plt.ylabel("Predicted Price ($100k)")
plt.legend()
plt.savefig('predicted_vs_actual.png')

"""
    To visualize the loss landscape for model 1, we create a grid of bias and weight values around the final parameters obtained from training.
    For each combination of bias and weight, we compute the predicted values and calculate the mean squared error loss.
    We then create a contour plot of the loss landscape, and overlay the path taken by the parameters during gradient descent optimization.
    This visualization helps us understand how the model converged to the optimal parameters and the shape of the loss function around those parameters.
    We also create a plot to compare the effect of different learning rates on the convergence of the training process.
"""
#  Extract the scalar float value from m1.bias if it's an array for linspace
b_val = float(m1.bias) if isinstance(m1.bias, np.ndarray) else m1.bias
b_range = np.linspace(b_val - 2, b_val + 2, 50)
w_range = np.linspace(m1.weights[0,0] - 2, m1.weights[0,0] + 2, 50)
B, W = np.meshgrid(b_range, w_range)
Z = np.zeros(B.shape)
for i in range(len(b_range)):
    for j in range(len(w_range)):
        y_p = X1_train * w_range[j] + b_range[i]
        Z[j, i] = (1 / (2 * len(y_train))) * np.sum((y_p - y_train)**2)





plt.figure(figsize=(8, 6))
plt.contour(B, W, Z, levels=30)
path = np.array(m1.theta_history)
plt.plot(path[:, 0], path[:, 1], 'r-o', markersize=3, label='GD Path')
plt.title("Model 1: Loss Contour and GD Path")
plt.xlabel("Bias (θ0)")
plt.ylabel("Weight (θ1)")
plt.legend()
plt.savefig('plot3.png')

rates = [0.001, 0.005, 0.01, 0.05]
colors_lr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
plt.figure(figsize=(10, 5))
for r, c in zip(rates, colors_lr):
    temp_model = linearRegression(learning_rate=r, iterations=200)
    temp_model.fit(X1_train, y_train)
    plt.plot(temp_model.loss_history, label=f'η = {r}', color=c)
    
plt.title("Effect of Learning Rates on Convergence")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()




#  Added annotation notes 
plt.annotate("η=0.05: fastest convergence\nbut risks instability",
             xy=(10, plt.gca().get_ylim()[1]*0.85), fontsize=9, color='#d62728')
plt.annotate("η=0.001: slowest,\nstill converging at epoch 200",
             xy=(120, plt.gca().get_ylim()[1]*0.6), fontsize=9, color='#1f77b4')
plt.savefig('plot4.png')
plt.show()

#  Include the print notes on learning rate performance 
print("\nObservation on learning rates:")
print("  η=0.001 — very slow convergence; loss still dropping at epoch 200.")
print("  η=0.005 — moderate; converges but needs many epochs.")
print("  η=0.01  — good balance of speed and stability.")
print("  η=0.05  — fastest convergence; may overshoot if too large for a dataset.")