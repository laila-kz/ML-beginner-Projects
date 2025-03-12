#we will use a regressor model since we want to predict a continuous value
#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn.datasets 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#importing the california house price dataset
house_price_dataset = sklearn.datasets.fetch_california_housing()

print(house_price_dataset)

#loading the dataset to a pandas dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

#add the target (price) column to the dataframe
house_price_dataframe['price']= house_price_dataset.target

#checking the first 5 rows of the dataframe
print(house_price_dataframe.head()) 

#checking the number of rows and columns in the dataframe
print(house_price_dataframe.shape) #(20640, 9)

#checking for missing values
print(house_price_dataframe.isnull().sum())

#statistical measures of the dataset
house_price_dataframe.describe()  #mean, std, min , max,....

#understanding correlation between various features in the dataset
# 1 --> positive correlation 
# 2 --> negative correlation 
#correlation represents the relationship between two variables/ features
#how one variable changes in relation to another
#used to identify the most relevant features for a model
#If two features are highly correlated, one might be redundant and can be removed.
correlation = house_price_dataframe.corr()

#If house price is strongly correlated with median income, it suggests income is an important predictor.

#constructing a heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar= True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='pink')
plt.show()

#splitting the data and labels/targets
X= house_price_dataframe.drop(['price'],axis=1) #droping a column axis=1 if you re droping a row axis=0
Y= house_price_dataframe['price']

print(X)
print(Y)

#splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
#random_state is used to fix the split. If you don't use it, you will get different splits every time you run the code.

print(X.shape, X_train.shape, X_test.shape) #(20640, 8) (16512, 8) (4128, 8)

#train the model
#loading the model
model = XGBRegressor()

#training the model with X_train
model.fit(X_train, Y_train)

#evaluation
#we cannot use accuracy score for regression models, basically what we do is we will count the number of the correct predictions and the original values and substract them
#we cant do it in regression since we are predicting continuous values 

#prediction on training data
#mean absolute error, mean squared error, root mean squared error


#accuracy for prediction on training data
training_data_prediction = model.predict(X_train)
#the model will predict the price for the training data and store it in training_data_prediction
#we will compare the predicted values with the actual values/ Y_train

#R squared error
score_1= metrics.r2_score(Y_train, training_data_prediction) # finds the variance between the two parameters

#mean absolue error
score_2= metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error: ", score_1) #closer to 1 the better
print("Mean Absolute Error: ", score_2) #closer to 0 the better

test_data_prediction= model.predict(X_test) #predicting the price for the test data
 
#finding the error value for test data
#R squared error
score_1_T= metrics.r2_score(Y_test, test_data_prediction) # finds the variance between the two parameters

#mean absolue error
score_2_T= metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error: ", score_1_T) #closer to 1 the better
print("Mean Absolute Error: ", score_2_T) #closer to 0 the better

#visualizing the actual prices and predicted prices
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()        


#we can see that the model is performing well on the training data but not on the test data


new_house= [[4.1, 0.4, 1.2, 0.5, 4.2, 3.2, 2.1, 2.2]]
new_house_array= np.asarray(new_house)
new_house_prediction= model.predict(new_house_array)
print(f"The predicted price of the new house is: ${new_house_prediction[0]*100000:.2f}") #since the price is in 100000s