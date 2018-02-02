from __future__ import print_function
from sklearn.svm import SVC
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

file = open(os.path.join('imbalanced_svm.csv'), mode='a')
file.write("ratio, split point, kernel, penalty param, gamma, fit time, train time, test time, train mse, test mse, train mae, test mae, accuracy, precision, recall, cv1, cv2, cv3, cv4, cv5, cv6, cv7, cv8, cv9, cv10\n")


def runSvm(ker,cee,gam, X_train,X_test, y_train, y_test,ratio,split):
    print("_______________________________________________________")
    log("Kernel:", ker, "C:", cee, "Gamma:", gam)
    print("ratio", ratio, "Time:", datetime.datetime.now(), "Kernel:", ker, "C:", cee, "Gamma:", gam)

    svm = SVC(kernel=ker, C=cee, gamma=gam, max_iter=1000000000)
    start = time.time()
    svm.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start

    start = time.time()
    y_pred_train = svm.predict(X_train)
    end = time.time()
    train_time = end-start

    start = time.time()
    y_pred_test = svm.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    accuracy = accuracy_score(y_train, y_pred_train)
    weighted_precison = precision_score(y_train, y_pred_train)
    weighted_recall = recall_score(y_train, y_pred_train)
    cv_score = cross_val_score(svm, X_test, y_test, cv = 10)
    
    
    file.write("SVC"+","+str(ratio)+","+str(split)+","+ker+","+str(c)+","+str(gam)+","+str(fit_time)+","+str(train_time)+","+str(test_time)
        +","+str(train_mse)+","+str(test_mse)+","+str(train_mae)+","+str(test_mae)+","
        +str(accuracy)+","+str(weighted_precison)+","+str(weighted_recall)+","
        +str(cv_score[0])+","+str(cv_score[1])+","+str(cv_score[2])+","+str(cv_score[3])+","+str(cv_score[4])+","
        +str(cv_score[5])+","+str(cv_score[6])+","+str(cv_score[7])+","+str(cv_score[8])+","+str(cv_score[9])+"\n")

def runNuSvm(nu, ker,d,gam, X_train,X_test, y_train, y_test,ratio):
    print("_______________________________________________________")
    log("Kernel:", ker, "C:", cee, "Gamma:", gam)
    print("ratio", ratio, "Time:", datetime.datetime.now(), "Kernel:", ker, "C:", cee, "Gamma:", gam)

    svm = NuSVC(nu =nu, kernel=ker, degree=d, gamma=gam, max_iter=1000000000)
    start = time.time()
    svm.fit(X_train, y_train)
    end = time.time()
    fit_time = end-start

    start = time.time()
    y_pred_train = svm.predict(X_train)
    end = time.time()
    train_time = end-start

    start = time.time()
    y_pred_test = svm.predict(X_test)
    end = time.time()
    test_time = end-start

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    accuracy = accuracy_score(y_train, y_pred_train)
    weighted_precison = precision_score(y_train, y_pred_train)
    weighted_recall = recall_score(y_train, y_pred_train)
    cv_score = cross_val_score(svm, X_test, y_test, cv = 10)
    
    
    file.write("NuSvm"+","+str(ratio)+","+str(nu)+","+ker+","+str(d)+","+str(gam)+","+str(fit_time)+","+str(train_time)+","+str(test_time)
        +","+str(train_mse)+","+str(test_mse)+","+str(train_mae)+","+str(test_mae)+","
        +str(accuracy)+","+str(weighted_precison)+","+str(weighted_recall)+","
        +str(cv_score[0])+","+str(cv_score[1])+","+str(cv_score[2])+","+str(cv_score[3])+","+str(cv_score[4])+","
        +str(cv_score[5])+","+str(cv_score[6])+","+str(cv_score[7])+","+str(cv_score[8])+","+str(cv_score[9])+"\n")


kernels = ['linear', 'rbf', 'poly']
c_variations= [1,2]

gamma_variations = [0.1,10]

splits = [0.2, 0.4]

nus = [0.3, 0.5]

degrees = [2,3,4]

# Regressing in 1D
# comment all plt.* for real regression

ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
n = 45212
# ,columns = ['age','job','martial','education','default','balance','housing','loan'
#                     ,'contact','day','month','duration','campaign','pdays','previous','poutcome','y'])

for r in ratios:
    name = 'Resampled_'+ str(r) + '.csv'
    df = pd.read_csv(name,header= None)
    arr = np.array(df)
    data = arr[1:,:]
    header = arr[0,:]
    df = pd.DataFrame(data = data, columns=header)
    X = np.array(df.drop(['y'], 1))
    y = np.array(df['y'].astype(float))
    for split in splits:
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split, random_state=0)
        for k in kernels:
            for c in c_variations:
                for gam in gamma_variations:
                    runSvm(k, c, gam, X_train, X_test, y_train, y_test, r,split)

                    if split == 0.8 and k == 'poly':
                        for d in degress: 
                            for nu in nus:
                                runNuSvm(nu, k, d, gam, X_train, X_test, y_train, y_test, r,split)

sys.stdout.close()
file.close()











