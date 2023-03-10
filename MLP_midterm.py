import os
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

plt.figure(figsize=(15,8))

X_data=[]
Y_data=[]
count=0
dir_path=r'/home/ws1/ML/DATASET2/'
for file_path in (sorted(os.listdir(dir_path))):
    path=dir_path+file_path+'/star.planet.forward'
    X_data.append(re.findall("\d+\.\d+", dir_path+file_path))
    
    file1 = open(path, 'r+')
    lines = file1.readlines()
    lines[10] = re.split(' |\t', lines[10])
    X_data[count]=[float(x) for x in X_data[count]]
    X_data[count].append(float(lines[10][1]))        # append the pressure of water in the atmosphere

    lines[11] = re.split(' |\t', lines[11])        # Train for the water pressure in the next iteration
    Y_data.append(float(lines[11][1]))
    count+=1
    file1.close()

print("data generated, now training the model")

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
acc=round(r2_score(Y_test, Y_pred), 4)
print('Accuracy of the model:  ', acc)

plt.subplot(2,3,1)
plt.plot(mlp.loss_curve_, 'r-.', label='Mantle potential temperature\nAccuracy = %f' %acc)
plt.xlabel('No. of iterations', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.legend(fontsize=13)

#####################################################################################################################################################


X_data=[]
Y_data=[]
count=0
dir_path=r'/home/ws1/ML/DATASET2/'
for file_path in (sorted(os.listdir(dir_path))):
    path=dir_path+file_path+'/star.planet.forward'
    X_data.append(re.findall("\d+\.\d+", dir_path+file_path))
    
    file1 = open(path, 'r+')
    lines = file1.readlines()
    lines[10] = re.split(' |\t', lines[10])
    X_data[count]=[float(x) for x in X_data[count]]
    X_data[count].append(float(lines[10][13]))        # append the pressure of water in the atmosphere

    lines[11] = re.split(' |\t', lines[11])        # Train for the water pressure in the next iteration
    Y_data.append(float(lines[11][13]))
    count+=1
    file1.close()

print("data generated, now training the model")

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
acc=round(r2_score(Y_test, Y_pred), 4)
print('Accuracy of the model:  ', acc)

plt.subplot(2,3,2)
plt.plot(mlp.loss_curve_, 'r-.', label='Net flux of atm\nAccuracy = %f' %acc)
plt.xlabel('No. of iterations', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.legend(fontsize=13)

#####################################################################################################################################################



X_data=[]
Y_data=[]
count=0
dir_path=r'/home/ws1/ML/DATASET2/'
for file_path in (sorted(os.listdir(dir_path))):
    path=dir_path+file_path+'/star.planet.forward'
    X_data.append(re.findall("\d+\.\d+", dir_path+file_path))
    
    file1 = open(path, 'r+')
    lines = file1.readlines()
    lines[10] = re.split(' |\t', lines[10])
    X_data[count]=[float(x) for x in X_data[count]]
    X_data[count].append(float(lines[10][8]))        # append the pressure of water in the atmosphere

    lines[11] = re.split(' |\t', lines[11])        # Train for the water pressure in the next iteration
    Y_data.append(float(lines[11][8]))
    count+=1
    file1.close()

print("data generated, now training the model")

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
acc=round(r2_score(Y_test, Y_pred), 4)
print('Accuracy of the model:  ', acc)

plt.subplot(2,3,3)
plt.plot(mlp.loss_curve_, 'r-.', label='Pressure of water in the atm\nAccuracy = %f' %acc)
plt.xlabel('No. of iterations', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.legend(fontsize=13)

#####################################################################################################################################################



X_data=[]
Y_data=[]
count=0
dir_path=r'/home/ws1/ML/DATASET2/'
for file_path in (sorted(os.listdir(dir_path))):
    path=dir_path+file_path+'/star.planet.forward'
    X_data.append(re.findall("\d+\.\d+", dir_path+file_path))
    
    file1 = open(path, 'r+')
    lines = file1.readlines()
    lines[10] = re.split(' |\t', lines[10])
    X_data[count]=[float(x) for x in X_data[count]]
    X_data[count].append(float(lines[10][4]))        # append the pressure of water in the atmosphere

    lines[11] = re.split(' |\t', lines[11])        # Train for the water pressure in the next iteration
    Y_data.append(float(lines[11][4]))
    count+=1
    file1.close()

print("data generated, now training the model")

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
acc=round(r2_score(Y_test, Y_pred), 4)
print('Accuracy of the model:  ', acc)

plt.subplot(2,3,4)
plt.plot(mlp.loss_curve_, 'r-.', label='$m_{H_2O}$ in MO and atm\nAccuracy = %f' %acc)
plt.xlabel('No. of iterations', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.legend(fontsize=13)

#####################################################################################################################################################



X_data=[]
Y_data=[]
count=0
dir_path=r'/home/ws1/ML/DATASET2/'
for file_path in (sorted(os.listdir(dir_path))):
    path=dir_path+file_path+'/star.planet.forward'
    X_data.append(re.findall("\d+\.\d+", dir_path+file_path))
    
    file1 = open(path, 'r+')
    lines = file1.readlines()
    lines[10] = re.split(' |\t', lines[10])
    X_data[count]=[float(x) for x in X_data[count]]
    X_data[count].append(float(lines[10][6]))        # append the pressure of water in the atmosphere

    lines[11] = re.split(' |\t', lines[11])        # Train for the water pressure in the next iteration
    Y_data.append(float(lines[11][6]))
    count+=1
    file1.close()

print("data generated, now training the model")

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
acc=round(r2_score(Y_test, Y_pred), 4)
print('Accuracy of the model:  ', acc)

plt.subplot(2,3,5)
plt.plot(mlp.loss_curve_, 'r-.', label='$m_{O_2}$ in MO and atm\nAccuracy = %f' %acc)
plt.xlabel('No. of iterations', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.legend(fontsize=13)

#####################################################################################################################################################



X_data=[]
Y_data=[]
count=0
dir_path=r'/home/ws1/ML/DATASET2/'
for file_path in (sorted(os.listdir(dir_path))):
    path=dir_path+file_path+'/star.planet.forward'
    X_data.append(re.findall("\d+\.\d+", dir_path+file_path))
    
    file1 = open(path, 'r+')
    lines = file1.readlines()
    lines[10] = re.split(' |\t', lines[10])
    X_data[count]=[float(x) for x in X_data[count]]
    X_data[count].append(float(lines[10][10]))        # append the pressure of water in the atmosphere

    lines[11] = re.split(' |\t', lines[11])        # Train for the water pressure in the next iteration
    Y_data.append(float(lines[11][10]))
    count+=1
    file1.close()

print("data generated, now training the model")

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
acc=round(r2_score(Y_test, Y_pred), 4)
print('Accuracy of the model:  ', acc)

plt.subplot(2,3,6)
plt.plot(mlp.loss_curve_, 'r-.', label='$m_{H_2}$ escaping from the atm\nAccuracy = %f' %acc)
plt.xlabel('No. of iterations', fontsize=13)
plt.ylabel('Loss', fontsize=13)
plt.legend(fontsize=13)
plt.savefig('loss2.png')
#plt.show()
#####################################################################################################################################################


