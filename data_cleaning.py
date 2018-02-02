import numpy as np
from sklearn import preprocessing, neighbors
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import pandas as pd

df = pd.read_csv('bank-full.csv')

df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        # convert Categorical var to number 
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

df.to_csv('bank_full_new.csv')

arr = np.array(df).astype(float)

arr = preprocessing.scale(arr)

df = pd.DataFrame(arr)

df.to_csv('normalized_bank_full_new.csv')

# X = np.array(df.drop(['y'], 1).astype(float))
# X = preprocessing.scale(X)

# y = np.array(df['y'])

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# KNN = neighbors.KNeighborsClassifier()
# KNN.fit(X_train, y_train)
# KNN_scores = cross_val_score(KNN, X_test, y_test, cv = 10)
# print("KNN Accuracy: %0.4f (+/- %0.2f)" % (KNN_scores.mean(), KNN_scores.std() * 2))

# svm1= svm.SVC()
# svm1.fit(X_train, y_train)
# svm1_scores = cross_val_score(svm1, X_test, y_test, cv = 10)
# print("SVM_SVC no kernel Accuracy: %0.4f (+/- %0.2f)" % (svm1_scores.mean(), svm1_scores.std() * 2))

# svm2= svm.SVC(kernel = 'linear')
# svm2.fit(X_train, y_train)
# svm2_scores = cross_val_score(svm2, X_test, y_test, cv = 10)
# print("SVM_SVC linear kernel Accuracy: %0.4f (+/- %0.2f)" % (svm2_scores.mean(), svm2_scores.std() * 2))

# svm3= svm.SVC(kernel = 'rbf')
# svm3.fit(X_train, y_train)
# svm3_scores = cross_val_score(svm3, X_test, y_test, cv = 10)
# print("SVM_SVC rbf kernel Accuracy: %0.4f (+/- %0.2f)" % (svm3_scores.mean(), svm3_scores.std() * 2))


# lin_svm = svm.LinearSVC()
# lin_svm.fit(X_train, y_train)
# lin_svm_scores = cross_val_score(lin_svm, X_test, y_test, cv = 10)
# print("linear SVC Accuracy: %0.4f (+/- %0.2f)" % (lin_svm_scores.mean(), lin_svm_scores.std() * 2))

# MLP = MLPClassifier()
# MLP.fit(X_train, y_train)
# MLP_scores = cross_val_score(MLP, X_test, y_test, cv = 10)
# print("MLP default Accuracy: %0.4f (+/- %0.2f)" % (MLP_scores.mean(), MLP_scores.std() * 2))

# MLP1 = MLPClassifier(activation='tanh')
# MLP1.fit(X_train, y_train)
# MLP1_scores = cross_val_score(MLP1, X_test, y_test, cv = 10)
# print("MLP with activation= tanh Accuracy: %0.4f (+/- %0.2f)" % (MLP1_scores.mean(), MLP1_scores.std() * 2))

# MLP2 = MLPClassifier(hidden_layer_sizes = ())
# MLP.fit(X_train, y_train)
# MLP_scores = cross_val_score(MLP, X_test, y_test, cv = 10)
# print("MLP Accuracy: %0.4f (+/- %0.2f)" % (MLP_scores.mean(), MLP_scores.std() * 2))

# # sample = 















