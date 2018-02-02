import pandas as pd
import numpy as np
import random


sample_sizes = [50, 100, 500, 1000, 3000, 5000,8000,10000,30000]
n = 45212
for size in sample_sizes:
    skip = sorted(random.sample(range(n), n-size))
    df = pd.read_csv('normalized_bank_full_new.csv',index_col=0,skiprows = skip,header=None)
    name = 'bank_'+ str(size) + '.csv'
    df.to_csv(name, header=None)