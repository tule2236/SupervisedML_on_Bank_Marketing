from __future__ import print_function
from sklearn import preprocessing, neighbors
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np
import time, datetime, timeit
import csv
import pandas as pd 

df = pd.read_csv('bank-full.csv',index_col = 0)

df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))
    return df

df = handle_non_numerical_data(df)

sample_sizes = [50, 100, 500, 1000, 3000, 5000,8000,10000,30000,45211]


for size in sample_sizes:
    sample = df.loc[np.random.choice(df.index, size, replace=False)]
    X = np.array(sample.drop(['y'], 1).astype(float))
    X = preprocessing.scale(X)
    y = np.array(sample['y'])
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    # SVM
    with open('SVM_experiment.csv','w') as csvfile:
        svm_writer = csv.writer(csvfile)

        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        c_variations = [1,2,4,8,16]
        gamma_variations = [0.1,0.01,0.001,0.0001,0.00001]
        i= 1

        for ker in kernels:
            for c in c_variations:
                for gam in gamma_variations:

                    i += 1
                    svm_classifier = svm.SVC(C=c,kernel=ker,gamma=gam)

                    start = time.time()
                    svm_classifier.fit(X_train, y_train)
                    end= time.time()
                    fit_time = end-start

                    start = time.time()
                    y_pred_train = svm_classifier.predict(X_train)
                    end = time.time()
                    train_time = end-start

                    start = time.time()
                    y_pred_test = svm_classifier.predict(X_test)
                    end = time.time()
                    test_time = end-start

                    cv_score = cross_val_score(svm_classifier, X_test, y_test, cv = 10)
                    train_mse = mean_squared_error(y_train, y_pred_train)
                    test_mse = mean_squared_error(y_test, y_pred_test)
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)

                    accuracy = accuracy_score(y_train, y_pred_train)
                    weighted_precison = precision_score(y_train, y_pred_train,average = 'weighted')
                    unweighted_precison = precision_score(y_train, y_pred_train,average = 'macro')
                    weighted_recall = recall_score(y_train, y_pred_train,average = 'weighted')
                    unweighted_recall = recall_score(y_train, y_pred_train,average = 'macro')

                    
                    svm_writer.writerow(str(size)+ker+str(c)+str(gam)+str(fit_time)
                        +str(train_time)+str(test_time)+str(accuracy)+ str(weighted_precison)
                        +str(unweighted_precison)+str(weighted_recall)+str(unweighted_recall)
                        +str(train_mse)
                        +str(test_mse)+str(train_mae)+str(test_mae)+str(cv_score))

                    print('svm',i)

for size in sample_sizes:
    sample = df.loc[np.random.choice(df.index, size, replace=False)]
    X = np.array(sample.drop(['y'], 1).astype(float))
    X = preprocessing.scale(X)
    y = np.array(sample['y'])
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15)
    # MLP
    with open('MLP_experiment.csv','w',newline='') as csvfile:
        mlp_writer = csv.writer(csvfile)

        activations = ['identity', 'logistic', 'tanh', 'relu']
        solvers = ['lbfgs', 'sgd', 'adam']
        learning_rates = ['constant','invscaling','adaptive']
        hidden_layer_sizes = [(50,),(100,),(500,),(50,50),(100,100),(500,500),
                            (50,50,50),(100,100,100),(500,500,500),(500,500,500,500)]
        j =1 
        for layer_size in hidden_layer_sizes:
            for a in activations:
                for s in solvers:
                    for l in learning_rates:
                        j +=1
                        MLP = MLPClassifier(hidden_layer_sizes=size,activation=a,solver=s,learning_rate=l)
                        
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

                        cv_score = cross_val_score(MLP, X_test, y_test, cv = 10)
                        train_mse = mean_squared_error(y_train, y_pred_train)
                        test_mse = mean_squared_error(y_test, y_pred_test)
                        train_mae = mean_absolute_error(y_train, y_pred_train)
                        test_mae = mean_absolute_error(y_test, y_pred_test)

                        accuracy = accuracy_score(y_train, y_pred_train)
                        weighted_precison = precision_score(y_train, y_pred_train,average = 'weighted')
                        unweighted_precison = precision_score(y_train, y_pred_train,average = 'macro')
                        weighted_recall = recall_score(y_train, y_pred_train,average = 'weighted')
                        unweighted_recall = recall_score(y_train, y_pred_train,average = 'macro')

                        mlp_writer.writerow(str(size)+str(layer_size)+a+s+str(l)+str(fit_time)
                        +str(train_time)+str(test_time)+str(accuracy)+ str(weighted_precison)
                        +str(unweighted_precison)+str(weighted_recall)+str(unweighted_recall)
                        +str(train_mse)+str(test_mse)+str(train_mae)+str(test_mae))

                        print('mlp',j)



























                

