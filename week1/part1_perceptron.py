#    ----- WEEK 1 part 1 -----
import pandas as pd # Data handling & manipulation.
import matplotlib.pyplot as plt # Creating Data visualizations.
import numpy as np # Numerical computing.
import seaborn as sns # Creating Data visualizations.Build on top of matplotlib.
from sklearn.datasets import load_diabetes # skit-learn constains numerous datasets for testing and learning purposes.

dataset=load_diabetes() 


""" Dataframe is a 2D data structure in pandas that can hold data of different types (like integers, floats, strings, etc.) in columns. 
 It is similar to a table in a relational database or an Excel spreadsheet. Each column in a DataFrame can be thought of as a Series, and the entire DataFrame can be thought of as a collection of Series objects. 
 The 'columns' parameter is used to specify the names of the columns in the DataFrame, which are taken from the 'feature_names' attribute of the dataset."""

df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['diabetic']=dataset.target # dataset.target contains the target variable.
print(df.columns) # prints ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6','diabetic'] -- 10 feature input and 1 label ('diabetic') variable.
X=df.drop(columns='diabetic') 
y=df['diabetic']    



# Why to user a threshold?
# Our perceptron model is made of only one neuron and it adjusts its weights and bias to find better decision boundary.
# It outputs in binary values 0 & 1.SO we are using threshold to convert the continuous output of our model into binary output.

threshold=np.median(y) 
y_binary=np.where(y>threshold,1,0) # converting continuous target variable into binary variable based on the threshold value. 
y=pd.Series(y_binary)
print(y)


# 
sns.pairplot(df,hue='diabetic')
plt.show()


# splitting the dataset into training and testing sets.
N=len(X) # size of dataset
train_size=int(0.8*N) # 80% of N (Size of dataset) used to train,rest to test the model.
np.random.seed(42) # Fix randomness -> it generates same variables every time.
indices=np.random.permutation(N) #Dataset may be ordered, which is very commom ,that is the reason behind shuffling the data using random premutation.
train_indicies=indices[:train_size] 
test_indicies=indices[train_size:]



# using iloc to split the dataset..
X_train=X.iloc[train_indicies]
X_test=X.iloc[test_indicies]
y_train=y.iloc[train_indicies]
y_test=y.iloc[test_indicies]



weights=np.random.randn(10,1) # I/P features are 10 so weights(10x1) are intilized with 10 random values..
# print(weights)                          
bias=np.random.randn(1,1) #bias allows model to fit data better.



# traning the model on  X_train(80% of dataset).
for epoch in range(100):
    for i in range(len(X_train)):
        x_i=X_train.iloc[i].values.reshape(-1,1) # .reshape this [5.1, 3.5, 1.4, 0.2] to [[5.1],[3.5],[1.4],[0.2]]..we need to reshape it because we want to perform matrix multiplication with weights .
        y_i=y_train.iloc[i] # y_i is the actual label. 

        """ predictions is output given by our model for the input x_i.
         we take transpose of weights matrix(1x10) and then do dot product with input features (10x1).
        It give (1x1) result and then we add bias to it to get final prediction."""

        prediction=(weights.T.dot(x_i) + bias).item()


        # error is the difference between actual label and predicted label. It gives us an idea about how well our model is performing.
        error=y_i-prediction 
        

        # precdiction less than or equal to 0 means predictions is wrong and we need to update weights and bias towards better decision boundary.
        if prediction*y_i<=0:

            # to not make weights and bias change drastically,we multiptly it with 0.001;
            weights=weights+y_i*x_i*0.001 
            bias+=y_i*0.001

    if epoch%25==0: # after every 25 epochs we are printing loss value to see how our model is learning.
        print(f"loss: {error}")




# Now testing is done on X_test(20% of dataset).This dataset is never shown to our model.
predictions=[]
for i in range(len(X_test)):
    prediction=(weights.T.dot(X_test.iloc[i].values.reshape(-1,1))+bias).item()
    if prediction>=threshold:  # if prediction is geater than certain limit,calculate it as 1 else 0;

        # if the prediction is greater than or equal to threshold, we consider it diabetic(positve).
        predictions.append(1)
    else:
        predictions.append(0)



# Calculating the confix matrix;
y_true=y_test.values
y_pred=np.array(predictions)
print(y_true)
print(predictions)
TP=np.sum((y_pred==1)&(y_true==1)) # TP(True Positive) :where our model predicted 1 and it is actually 1. 
TN=np.sum((y_pred==0)&(y_true==0)) # TN(True Negative) :where our model predicted 0 and it is actually 0. 
FP=np.sum((y_pred==1)&(y_true==0)) # FP(False Positive) :where our model predicted 1 and it is actually 0. 
FN=np.sum((y_pred==0)&(y_true==1)) # FN(False Negative) :where our model predicted 0 and it is actually 1. 



# drawing the confusion matrix using matplotlib.
conf_matrix=np.array([[TP,FP],[FN,TN]])
print(conf_matrix)

plt.imshow(conf_matrix)
plt.colorbar()

plt.xticks([0,1],["Pred+","Pred-"])
plt.yticks([0,1],["Actual+","Actual-"])

for i in range(2):
    for j in range(2):
        plt.text(j,i,conf_matrix[i,j],
                ha="center", va="center", color="white")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()