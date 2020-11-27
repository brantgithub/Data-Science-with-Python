### 1. Import Packages --------------------------------------------

import pandas as pd  

import warnings
warnings.filterwarnings('ignore')   # in the process of k-fold cross validation, there maybe warnings


### 2. Data Preparation ------------------------------------------

train = pd.read_csv('C:/Users/brant/Google Drive/extra study/Python/training data set.csv', header = None)

train = train[0].str.split(',', expand=True)    #slipt by comma

column_names = ["time" + str(x) for x in range(1,501)]
column_names.append("index")

train.columns = column_names

## split into train and test datasets

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(train.drop('index', axis=1),
                                                   train['index'], test_size=0.2,
                                                   random_state=42)


### 3. Linear Regression -----------------------------------------------------

from sklearn.linear_model import LinearRegression

# call the model and fit the data

LinReg = LinearRegression()
LinReg.fit(X_train, y_train)

y_pred_linreg = LinReg.predict(X_test)

# Define a function to give final_score and accuracy

def score_accuracy(y_test, y_pred):
    y_pred_sr = pd.Series(y_pred)
    y_test_sr = y_test.reset_index(drop = True)
    y_test_sr
    
    # Compare test and pred
    compare_pred_test = pd.DataFrame({ 'test': y_test_sr, 'pred': y_pred_sr} ) 
    compare_pred_test['test'] = compare_pred_test['test'].astype(float)
    compare_pred_test['diff'] = abs((compare_pred_test['pred'] - compare_pred_test['test'])/compare_pred_test['test'])

    final_score = len(compare_pred_test[compare_pred_test['diff']<0.002].index) / len(compare_pred_test.index)
    
    accuracy = 1- compare_pred_test['diff'].mean()
    
    print(final_score, accuracy)

score_accuracy(y_test, y_pred_linreg) #1.0, 99.9813%


### 4. Multiple Layer Perception ---------------------------------------------

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV    # for K-Fold Cross Validation

# define a function to print K-Fold Cross-Validation results
def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

# Hyperparamters Tuning

mlp = MLPRegressor()
parameters = {
    'hidden_layer_sizes': [(50,), (100,),(150)],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

cv = GridSearchCV(mlp, parameters, cv=2)
cv.fit(X_train, y_train)

print_results(cv) ### best parameter --- hidden_layer_sizes = 150, learning_rate = 'adaptive'

# call the model and fit the data

mlp = MLPRegressor(hidden_layer_sizes = 150, learning_rate = 'adaptive')
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
score_accuracy(y_test, y_pred_mlp) #0.98, 99.913%


### 5. Random Forest -------------------------

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
parameters = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10]
}

cv = GridSearchCV(rf, parameters, cv=2)
cv.fit(X_train, y_train)

print_results(cv)  ### 'max_depth': None, 'n_estimators': 150


# call the model and fit the data

rf = RandomForestRegressor(max_depth= None, n_estimators= 150)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
score_accuracy(y_test, y_pred_rf)    #1.0, 99.9819%


### 6. Output -----------------------------

# the best model is Rondom Forest with max_depth= None, n_estimators= 150

given_test = pd.read_csv('C:/Users/brant/Google Drive/extra study/Python/test data set.csv', header = None)
given_test = given_test[0].str.split(',', expand=True)

y_pred_final = rf.predict(given_test)
y_pred_final = pd.DataFrame(data = y_pred_final)
y_pred_final.columns=['Predicted S&P Index']

y_pred_final.to_csv('prediction.csv', header = None, index = None)
