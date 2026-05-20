#   ----- WEEK 1 part 2 -----
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.datasets import load_diabetes
import math

""" 
Dataframe is a 2D data structure in pandas that can hold data of different types (like integers, floats, strings, etc.) in columns. 
It is similar to a table in a relational database or an Excel spreadsheet. Each column in a DataFrame can be thought of as a Series, and the entire DataFrame can be thought of as a collection of Series objects. 
The 'columns' parameter is used to specify the names of the columns in the DataFrame, which are taken from the 'feature_names' attribute of the dataset.
"""
dataset=load_diabetes()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['daibetic']=dataset.target

X=df.drop(columns='daibetic') 
y=df['daibetic']

"""
 Why to user a threshold?
 Our perceptron model is made of only one neuron and it adjusts its weights and bias to find better decision boundary.
 It outputs in binary values 0 & 1.SO we are using threshold to convert the continuous output of our model into binary output.
"""
threshold=np.median(y) 
y_binary=np.where(y>threshold,1,0) # converting continuous target variable into binary variable based on the threshold value. 
y=pd.Series(y_binary)
print(y)

# Splitting the dataset into training and testing sets.
# In this way we make sure that our model is evaluated on unseen data, which gives us a better estimate of its performance in real-world scenarios.
train_size=int(0.8*len(X))

# it fixes the randomness of our data splitting process, ensuring that we get the same training and testing sets every time we run the code.
# This is important for reproducibility and debugging purposes.
np.random.seed(42)
indicies=np.random.permutation(len(X))

# Splitting the dataset into training and testing sets using the shuffled indices.
train_indicies=indicies[:train_size]
test_indicies=indicies[train_size:]

X_train=X.iloc[train_indicies]
X_test=X.iloc[test_indicies]
y_train=y.iloc[train_indicies]
y_test=y.iloc[test_indicies]

# It squashes any real number into (0,1) so output = probability.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary Cross-Entropy loss function for logistic regression.
# BCE = -1/m * [y*log(p) + (1-y)*log(1-p)]
def binary_cross_entropy(y_true, y_prob):
    eps = 1e-8   # tiny value to avoid log(0)
    return -np.mean(y_true * np.log(y_prob + eps) +
                    (1 - y_true) * np.log(1 - y_prob + eps))

""" 
defining logistic regression function --- 
 Input parameters:
 X: Input features for training the model.
 y: Target variable for training the model.
 learning_rate: A hyperparameter that controls the step size of parameter updates during gradient descent.
 epochs: The number of times the entire training dataset will be passed through the model during training.
 batch_size: The number of samples processed before the model's parameters are updated during training.
 """

def logisticRegression(X,y,learning_rate,epochs,batch_size): 
    X=np.array(X)
    y=np.array(y).reshape(-1, 1)
    m=len(X)

    """ 
    we will also include bias in theta matrix, that will increase the no of rows by 1.
     new dimension of theta matrix = no of input features + 1 (Another way to include bias nothing more)
     Multiplied by 0.01 for smaller initial weights to prevent early saturation of the sigmoid function.
    """
    theta=np.random.randn(11,1) * 0.01

    # we will add a column of 1's to Input features for bias term.
    X_bias=np.c_[np.ones((m,1)),X]

    cost_history=[]  # keep track of cost function.

    # Epochs is the number of times the entire training dataset will be passed through the model during training.
    for epoch in range(epochs):

        # It generates a random permutation of indices from 0 to m-1, which is used to shuffle the data at the beginning of each epoch. 
        # This helps in improving the convergence of the gradient descent algorithm by ensuring that the model does not learn from the data in a fixed order.
        indxs=np.random.permutation(m)          
        X_shuffled=X_bias[indxs]
        y_shuffled=y[indxs]

        """ 
        Inside the each epoch,we iterate through the entire shuffled training dataset.
         We process the data in batches of size 'batch_size' to update the model parameters (theta) using gradient descent.
         which means after every batch we are updating the theta values towards the direction of minimum cost function.
        """
        for i in range(0,m,batch_size):
            X_batch=X_shuffled[i:i+batch_size]
            y_batch=y_shuffled[i:i+batch_size]

            # calculating predicted probabilities using sigmoid function.
            probs = sigmoid(X_batch.dot(theta))
            
            # Corrected gradient calculation for Binary Cross-Entropy
            # The previous linear regression gradient formula is replaced with the logistic gradient.
            gradients = (1 / len(X_batch)) * X_batch.T.dot(probs - y_batch)
            
            """ updating theta values towards the direction of minimun cost function.
             learning_rate is a hyperparameter (usually a small positive value) that controls the step size of the parameter updates during gradient descent.
             If the learning rate is too large, the model may overshoot the optimal parameters and diverge, while if it is too small, the model may take a long time to converge.
             ANALOGY : Take the analogy of a monster rolling down a hill to find the lowest point.
                       IF the monster takes very large steps,he may jump from one hill to another and never reach the lowest point.
                       On the other hand, if he takes very small steps,he will take a long time to reach the lowest point."""
            
            theta=theta-learning_rate*gradients

        # calculating predictions after updating all the theta values -- for one epoch.
        # Must apply sigmoid to get probabilities for the cost function.
        predictions = sigmoid(X_bias.dot(theta))
        
        # calculating cost function using the properly defined binary_cross_entropy method.
        cost = binary_cross_entropy(y, predictions)
        cost_history.append(cost)
        
        # printing cost value after every 100 epochs to see how our model is learning.
        if epoch%100==0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")

    return theta,cost_history

# Notice the learning rate was slightly increased to allow the model to converge given the logistic gradient math
theta_final,cost_history=logisticRegression(X_train,y_train,learning_rate=0.1,epochs=1000,batch_size=64)

plt.plot(np.array(cost_history).flatten())
plt.xlabel('Epochs')
plt.ylabel('Cost (Binary Cross-Entropy)')
plt.title('Cost Function during Training')
plt.show()

print(f"Final parameters: \n{theta_final}\n")

# Evaluating the model
X_test=np.array(X_test)
y_test=np.array(y_test).reshape(-1,1)
m=len(X_test)
X_test_bias=np.c_[np.ones((m,1)), X_test]

# Convert testing predictions into probabilities, then into binary labels (0 or 1)
probs_test = sigmoid(X_test_bias.dot(theta_final))
y_pred = (probs_test >= 0.5).astype(int)

# Display accuracy and classification metrics rather than linear residual loss
accuracy = np.mean(y_pred == y_test)
print(f"Final Test Accuracy: {accuracy:.2%}")

# Visualizing classification results instead of a continuous scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_test)), y_test, label="Actual Class", alpha=0.6, marker='o')
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Class", alpha=0.6, marker='x')
plt.yticks([0, 1])
plt.xlabel('Sample Index')
plt.ylabel('Diabetic Classification (0 or 1)')
plt.title(f'Actual vs Predicted on Test Set (Accuracy: {accuracy:.2%})')
plt.legend()
plt.show()