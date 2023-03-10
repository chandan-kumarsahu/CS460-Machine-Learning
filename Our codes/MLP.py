import os
import re
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_data=[]
Y_data=[]
count=0
dir_path=r'/home/ws1/ML/DATASET2/'
for file_path in tqdm(sorted(os.listdir(dir_path))):
    path=dir_path+file_path+'/star.planet.forward'
    X_data.append(re.findall("\d+\.\d+", dir_path+file_path))
    
    file1 = open(path, 'r+')
    lines = file1.readlines()
    lines[30] = re.split(' |\t', lines[30])
    X_data[count]=[float(x) for x in X_data[count]]
    X_data[count].append(float(lines[30][8]))        # append the pressure of water in the atmosphere

    lines[31] = re.split(' |\t', lines[31])        # Train for the water pressure in the next iteration
    Y_data.append(float(lines[31][8]))
    count+=1
    file1.close()

print("data generated, now training the model")

#scaler = StandardScaler()
#X_data = scaler.fit_transform(X_data)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

# Create an MLP regressor with two hidden layers of 16 neurons each
mlp = MLPRegressor(hidden_layer_sizes=(16, 16, 16), activation='relu', solver='adam', alpha=0.001, max_iter=10000, tol=1e-4, random_state=42)

# Train the MLP on the training data
mlp.fit(X_train, Y_train)

# Use the trained MLP to make predictions on the testing data
Y_pred = mlp.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:     ", mse)

# Calculating the accuracy of the model
from sklearn.metrics import r2_score
print('Accuracy of the model:  ', r2_score(Y_test, Y_pred))

plt.plot(Y_test, Y_pred, 'r.')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

################################################################################################

# plot the learning curve, with x axis as epochs and y axis as the loss
plt.plot(mlp.loss_curve_, 'r-.')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
