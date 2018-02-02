import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA

df = pd.read_csv('bank_full_new.csv')
arr = df.values
X = arr[:,:16]
y = arr[:, 16]
# change n_components to different number to explore the effect 
# of attributes on algorithm performance
pca = PCA(n_components = 7)
pca.fit(X)
X7 = pca.transform(X)

df = pd.DataFrame(X7)

df.to_csv('X3_45212.csv')