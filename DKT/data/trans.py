import numpy as np
import pandas as pd

with open("storage", 'r', encoding='utf-8') as f:
    value = f.read()
    storage = eval(value)

ques = pd.read_csv("part_cons_train.csv")
ques_key = {}

for i in range(len(ques)):
    u = ques['user_id'][i]
    s = ques['skill_id'][i]
    if ques_key.get(u) is None:
        ques_key[u] = [s]
    else:
        ques_key[u].append(s)

for key, value in storage.items():
    if (len(value) + 1 == len(ques_key[int(key)])):
        ques_key[int(key)] = ques_key[int(key)][1:]
    if (len(value) + 2 == len(ques_key[int(key)])):
        ques_key[int(key)] = ques_key[int(key)][1:300] + \
            ques_key[int(key)][301:]

temp = []

for key, value in storage.items():
    tt = ques_key[int(key)]
    for i in range(len(value)):
        storage[key][i] = [tt[i]] + storage[key][i]

datas = []
for key, value in storage.items():
    for v in value:
        datas.append([key] + v)

result = pd.DataFrame(columns=['user_id', 'bot'] +
                      list(range(116)), data=datas)
result.to_csv("train.csv", header=None, index=None)

data = pd.read_csv("train.csv", header=None)
users = data[0].value_counts()[:50].index.tolist()
data = data[data[0].isin(users)]
data[0] = 2587

data.to_csv("fifty_person_train.csv", header=None, index=None)
