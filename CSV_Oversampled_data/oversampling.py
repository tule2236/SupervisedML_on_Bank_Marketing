# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, SMOTE
import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('bank_full_new.csv')
X = np.array(df.drop(['y'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['y'])

# Generate the dataset


# Apply the random over-sampling
# ros = RandomOverSampler()
# X_resampled, y_resampled = ros.fit_sample(X, y)
# print(len(X_resampled))
# print(len(y_resampled))
# # y_resampled = np.transpose(y_resampled)
# X_resampled = pd.DataFrame(X_resampled)
# y_resampled = pd.DataFrame(y_resampled)

# X_resampled.to_csv('oversampled_bannk.csv')
# y_resampled.to_csv('y.csv')

# print('Original dataset shape {}'.format(Counter(y)))

ratios = [0.8]
for r in ratios:
  sm = SMOTE(random_state=42,ratio=r)
  X_res, y_res = sm.fit_sample(X, y)
  print('Resampled dataset shape {}'.format(Counter(y_res)))
  X_resampled = pd.DataFrame(X_res)
  y_resampled = pd.DataFrame(y_res)
  name = 'Resampled_'+str(r)+'.csv'
  X_resampled.to_csv(name)
  y_name = name = 'ResampledLabel_'+str(r)+'.csv'
  y_resampled.to_csv(y_name)
