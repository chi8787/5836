import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from colorama import Fore, Back , Style
import statistics
from statistics import stdev
with open('abalone.txt', 'r') as file:
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    data = pd.read_csv(file, names=column_names, delimiter=',')
#data processing
#1  
data['Sex'] = data['Sex'].map({'M' : 0, 'F' : 1, 'I' : 2})    # I = 0 OR 1 ?? -->2

#2
correlation_map_matrix = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_map_matrix, annot = True, cmap = 'Spectral_r', linewidths = 0.5)
plt.title('correlation heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

#3
p_feature, n_feature = 'Shell weight', 'Sex'
plt.figure(figsize=(8,5))

     #positive
plt.subplot(1,2,1)
plt.scatter(data[p_feature], data['Rings'], color = 'blue', label = p_feature)
plt.title('Shell weight - Ring Age')
plt.xlabel('Shell weight')
plt.ylabel('Ring Age')

     #negative
plt.subplot(1,2,2)
plt.scatter(data[n_feature], data['Rings'], color = 'red', label = n_feature)
plt.title('Viscera weight - Ring Age')
plt.xlabel('Sex')
plt.ylabel('Ring Age')
plt.tight_layout(pad = 4)
plt.savefig('Viscera_weight-Ring_Age.png')
plt.show()

#4
plt.figure(figsize = (12, 5))
 # first: Shell weight
plt.subplot(1,3,1)
sns.histplot(data['Shell weight'], bins = 20, kde = True)
plt.title('Shell weight')

 # second: Whole weight
plt.subplot(1,3,2)
sns.histplot(data['Whole weight'], bins = 20, kde = True)
plt.title('Whole weight')

 # third: Rings
plt.subplot(1,3,3)
sns.histplot(data['Rings'], bins = 20, kde = True)
plt.title('Rings Age')

plt.tight_layout()
plt.savefig('Rings_Age.png')
plt.show()

#5
x = data.drop('Rings', axis=1)
y = data['Rings']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=88)


#Modelling
experiments = 3
#1
RMSE_1 = []   # create list to calculate the sum
R_squared_score_1 = []
for i in range(experiments):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=88 + i)  #use different random_state to represent three different test and train dataset
    regression = LinearRegression()  #LinearRegression
    regression.fit(x_train, y_train)
    y_pred = regression.predict(x_test)    # predict y value
    RMSE_1.append(np.sqrt(mean_squared_error(y_test, y_pred)))   
    R_squared_score_1.append(r2_score(y_test, y_pred))
a = sum(RMSE_1) / experiments    # close to 0 is better
b = sum(R_squared_score_1) / experiments    # close to 1 is better

print(f'RMSE: {a}, std = stdev(RMSE_1)')
print(f'R_squared_score: {b}, std = stdev(R_squared_score_1)')
plt.figure(figsize= (10,6))
plt.scatter(y_test, y_pred)
plt.title('y_test  VS  y_pred')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.grid(True)
plt.savefig('y_test.VS.y_pred.png')
plt.show()

#2
RMSE_nor = []
R_squared_score_nor = []

for i in range(experiments):   # split 6 and 4 for train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=88 + i)   #use different random_state to represent three different test and train dataset
    scaler = MinMaxScaler()   # Normalization would cause problem
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    normalization = LinearRegression()
    normalization.fit(x_train_scaled, y_train)  # normalize the training data
    y_pred_normalize = normalization.predict(x_test_scaled)
    RMSE_nor.append(np.sqrt(mean_squared_error(y_test, y_pred_normalize)))
    R_squared_score_nor.append(r2_score(y_test, y_pred_normalize))
    
c = sum(RMSE_nor) / experiments
d = sum(R_squared_score_nor) / experiments
    
print(Back.RED + 'Normalization:')
print(Style.RESET_ALL)
print(f'RMSE: {a}', end = '     vs    ')
print(f'RMSE-normalization: {c}')
print(f'R_squared_score: {b}', end = '     vs    ')
print(f'R_squared_score-normalization: {d}\n')
print(f'RMSE-normalization std: {stdev(RMSE_nor)}')
print(f'R_squared_score-normalization std: {stdev(R_squared_score_nor)}')
#3
RMSE_sel = []
R_squared_score_sel = []
features = ['Shell weight', 'Diameter']


for i in range(experiments):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=88 + i)  #use different random_state to represent three different test and train dataset
    x_train_sel = x_train[features]
    x_test_sel = x_test[features]
    sel_regression = LinearRegression()
    sel_regression.fit(x_train_sel, y_train)
    y_pred_sel = sel_regression.predict(x_test_sel)
    RMSE_sel.append(np.sqrt(mean_squared_error(y_test, y_pred_sel)))
    R_squared_score_sel.append(r2_score(y_test, y_pred_sel))
e = sum(RMSE_sel) / experiments
f = sum(R_squared_score_sel) / experiments

print(Back.RED + f'Selected {features[0]} and {features[1]}:')
print(Style.RESET_ALL)
print(f'RMSE of {features[0]} and {features[1]}: {e}, std: {stdev(RMSE_sel)}')
print(f'R_squared_score of {features[0]} and {features[1]}: {f}\n, std: {stdev(R_squared_score_sel)}')

#4
# try
hidden_neurons_values = [10, 20, 30]
learning_rate_values = [0.001, 0.01, 0.1]

best_rmse = float('inf')
best_R_score = -float('inf')
best_layers = 0
best_learning_rate = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=88)
for _ in range(experiments):
    hidden_neurons_layers = np.random.choice(hidden_neurons_values)  # choose from (10, 20, 30)
    learning_rate = np.random.choice(learning_rate_values)  # choose from (0.001, 0.01, 0.1)
    model_nn = MLPRegressor(hidden_layer_sizes=(hidden_neurons_layers,), activation='relu', solver='sgd',
                            learning_rate_init=learning_rate, max_iter=1000, alpha=0.01)
    model_nn.fit(x_train_scaled, y_train)
    y_test_pred_nn = model_nn.predict(x_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_nn))
    R_score = r2_score(y_test, y_test_pred_nn)
    
    if rmse < best_rmse:   # change if new rmse is better than previous
        best_rmse = rmse
        best_R_score = R_score    # change r score
        best_layers = hidden_neurons_layers
        best_learning_rate = learning_rate
    # if R_score < best_R_score:    # use R_score as our compare target
    #     best_rmse = rmse
    #     best_R_score = R_score
    #     best_layers = hidden_neurons_layers
    #     best_learning_rate = learning_rate

print(Back.RED + "neural network:")
print(Style.RESET_ALL)
print(f"Hidden Neurons: {best_layers}")
print(f"Learning Rate: {best_learning_rate}")
print(f"RMSE: {best_rmse}")
print(f"R-squared: {best_R_score}")
