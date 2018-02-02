'''
k = 5
0.889748977109

k = 4
0.895167532898
[ 0.88611234  0.82905794  0.86551648  0.86352577  0.83676178  0.87215218
  0.86374696  0.86994028  0.89847379  0.75138244]

k=3
0.888311401084
[ 0.88876603  0.8014153   0.85755364  0.845167    0.79893829  0.85025437
  0.84383986  0.845167    0.87900907  0.69409423]

k = 2
0.895831029526
[ 0.88522778  0.8202123   0.86308339  0.86020792  0.83454988  0.86219863
  0.86551648  0.86374696  0.88409644  0.75801814]
'''

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

df = pd.read_csv('Sample Size/bank_45212.csv')
X = np.array(df.drop(['y'], 1).astype(float))
y = np.array(df['y'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

alphas = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
train_errors = list()
test_errors = list()
for k in alphas:
	Kclf = KNeighborsClassifier(k)
	Kclf.fit(X_train, y_train)
	train_error = 1 - Kclf.score(X_train, y_train)
	train_errors.append(train_error)
	test_error = 1 - Kclf.score(X_test, y_test)
	test_errors.append(test_error)


i_alpha_optim = np.argmax(test_errors)
alpha_optim = alphas[i_alpha_optim]
print("Optimal regularization parameter : %s" % alpha_optim)

# Estimate the coef_ on full data with optimal regularization parameter
# Kclf.set_params(k=alpha_optim)
# coef_ = enet.fit(X, y).coef_

plt.subplot(2, 1, 1)
plt.semilogx(alphas, train_errors, label='Train Error')
plt.semilogx(alphas, test_errors, label='Test Error')
plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([-0.1, 0.2])
plt.xlabel('K values')
plt.ylabel('Error')
plt.legend()
plt.show()

# # Show estimated coef_ vs true coef
# plt.subplot(2, 1, 2)
# plt.plot(coef, label='True coef')
# plt.plot(coef_, label='Estimated coef')

# plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)
# plt.show()

# cv = cross_val_score(Kclf, X, y, cv = 10)
# print(score)
# print(cv)











