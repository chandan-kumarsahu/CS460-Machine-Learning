import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

DATA=[]
LABEL=[]

dir_path=r'/home/ws1/ML/DATASET2/'

path=dir_path+'ms0.2_a0.06_mp4.0_rp1.855_wm2.0'+'/star.planet.forward'
file1 = open(path, 'r+')
lines = file1.readlines()
lines[0] = re.split(' |\t', lines[0])
lim=len(lines[0])

for i in tqdm(range(lim-1)):
    count=0
    X_data=[]
    Y_data=[]
    for file_path in sorted(os.listdir(dir_path)):
        path=dir_path+file_path+'/star.planet.forward'
        X_data.append(re.findall("\d+\.\d+", file_path))

        file1 = open(path, 'r+')
        lines = file1.readlines()
        lines[20] = re.split(' |\t', lines[20])
        X_data[count]=[float(x) for x in X_data[count]]
        X_data[count].append(float(lines[20][i]))        # append the mantle potential temperature

        lines[21] = re.split(' |\t', lines[21])        # Train for the mantle potential temperature in the next iteration
        Y_data.append(float(lines[21][i]))
        count+=1
        file1.close()
    X_data=np.array(X_data)
    Y_data=np.array(Y_data)
    DATA.append(X_data)
    LABEL.append(Y_data)

print(len(LABEL))
print(len(LABEL[0]))
#print((LABEL[0]))

print('Data saved successfully!')
print('\n\n')


Names=['Time', 'PotTemp', 'SurfTemp', 'SolidRadius', 'WaterMassMOAtm', 'WaterMassSol', 'OxygenMassMOAtm', 'OxygenMassSol', 'PressWaterAtm', 'PressOxygenAtm', 'HydrogenMassSpace', 'OxygenMassSpace', 'FracFe2O3Man', 'NetFluxAtmo', 'WaterFracMelt', 'RadioPower', 'TidalPower', 'SemiMajorAxis', 'HZInnerEdge']

Y_TEST=[]
Y_PRED=[]

for i in range(len(DATA)):
    print('Training for: ', Names[i], '################################################')
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(DATA[i], LABEL[i], test_size=0.2, random_state=100)

    # Create an MLP regressor with two hidden layers of 20 neurons each
    mlp = MLPRegressor(hidden_layer_sizes=(32, 32, 32, 32), activation='relu', solver='adam', alpha=0.001, max_iter=10000, tol=1e-6, random_state=42)

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
    print('\n')

    Y_TEST.append(Y_test)
    Y_PRED.append(Y_pred)

for i in range(len(Y_TEST)):
    plt.plot(Y_TEST[i], Y_PRED[i], 'r.')
    plt.title('True vs Predicted '+Names[i])
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()

#    # Save the model
#    from sklearn.externals import joblib
#    joblib.dump(mlp, 'MLP_PotTemp.pkl')


