from __future__ import print_function
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import matplotlib.pyplot as plt
import timeit
import numpy as np
import os, sys
import time, datetime
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import random

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

# df = pd.read_csv('normalized_bank_full_new.csv',index_col = 0)

# df.convert_objects(convert_numeric=True)
# df.fillna(0, inplace=True)

# def handle_non_numerical_data(df):
#     columns = df.columns.values
#     for column in columns:
#         text_digit_vals = {}
#         def convert_to_int(val):
#             return text_digit_vals[val]

#         if df[column].dtype != np.int64 and df[column].dtype != np.float64:
#             column_contents = df[column].values.tolist()
#             unique_elements = set(column_contents)
#             x = 0
#             for unique in unique_elements:
#                 if unique not in text_digit_vals:
#                     text_digit_vals[unique] = x
#                     x+=1
#             df[column] = list(map(convert_to_int, df[column]))
#     return df

# df = handle_non_numerical_data(df)

file = open(os.path.join('auto_ann.csv'), mode='a')
file.write("sample size, layer size, activation, solver, fit time, train time, test time, train mse, test mse, train mae, test mae, accuracy, precision, recall\n")


def runMLP(layer_size,a,s, X_train,X_test, y_train, y_test,sample_size):
    print("_______________________________________________________")
    log("layer_size", layer_size, "activation:", a, "solver:",s)
    print("sample size:" , sample_size,"Time:", datetime.datetime.now(), "layer_size", layer_size, "activation:", a, "solver:",s)

    MLP = MLPClassifier(hidden_layer_sizes=layer_size,activation=a,solver=s)

    start = time.time()
    MLP.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start
    start = time.time()
    y_pred_train = MLP.predict(X_train)
    end = time.time()
    train_time = end-start
    start = time.time()
    y_pred_test = MLP.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    accuracy = accuracy_score(y_train, y_pred_train)
    weighted_precison = precision_score(y_train, y_pred_train)
    weighted_recall = recall_score(y_train, y_pred_train)
    # cv_score = cross_val_score(svm, X_test, y_test, cv = 10)
    node_number = layer_size[0]
    layer_length = len(layer_size)
    
    file.write(str(sample_size)+","+str(layer_length)+","+str(node_number)+","+a+","+s+","+str(fit_time)+","+str(train_time)+","+str(test_time)
        +","+str(train_mse)+","+str(test_mse)+","+str(train_mae)+","+str(test_mae)+","
        +str(accuracy)+","+str(weighted_precison)+","+str(weighted_recall)+"\n")

activations = ['identity', 'logistic', 'tanh', 'relu']
solvers = ['lbfgs', 'sgd', 'adam']
hidden_layer_sizes = [(50,),(500,),(50,50),(500,500),
                (50,50,50),(500,500,500),(50,50,50,50),(500,500,500),(500,500,500,500,500)]

sample_sizes = [50, 100, 500, 1000, 3000, 5000,8000,10000,30000,45212]

for size in sample_sizes:
    name = 'bank_'+ str(size) + '.csv'
    df = pd.read_csv(name, index_col=0)
    # print(sample.head())
    X = np.array(df.drop(['y'], 1))
    y = np.array(df['y'])
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    for a in activations:
        for s in solvers:
            for l in hidden_layer_sizes:
                # skip = sorted(random.sample(range(n), n-size))
                # sample = pd.read_csv('normalized_bank_full_new.csv',index_col=0,skiprows = skip)
                    
                # print(sample.head())
                
                runMLP(l,a,s, X_train, X_test, y_train, y_test,size)
sys.stdout.close()
file.close()











