import pandas as pd
import numpy as np

data = pd.read_csv("DKT/data/fifty_person_train.csv", header=None)

temp_data = data.iloc[:, 2:]
df = temp_data.corr()
temp2 = df.values

for i in range(116):
    for j in range(116):
        if i == j:
            temp2[i, j] = 0
            continue
        if abs(temp2[i, j]) > 0.6:
            temp2[i, j] = 1
        else:
            temp2[i, j] = 0

np.save("result/adj_matrix.npy", temp2)
