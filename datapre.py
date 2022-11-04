import numpy as np
import pandas as pd

data_path = "data/checkins_lessons_checkouts_training.csv"

loaded_data = (
    pd.read_csv(data_path, index_col=False)
    .sort_values(["UserId", "QuizSessionId", "Timestamp"], axis=0)
    .reset_index(drop=True)
)

loaded_data = loaded_data.dropna(axis=0)
loaded_data = loaded_data.reset_index(drop=True)

_, idx, counts = np.unique(
    np.array(loaded_data["UserId"]), return_counts=True, return_index=True)
list_idx_to_remove_nest = [list(range(ind, ind + counts[counts < 10][c]))
                           for c, ind in enumerate(idx[counts < 10])]
list_idx_to_remove = [
    v for sublist in list_idx_to_remove_nest for v in sublist]  # flatten it
proc_loaded_data = loaded_data.drop(
    labels=list_idx_to_remove, axis=0).reset_index(drop=True)

data = proc_loaded_data[['UserId', 'ConstructId', 'IsCorrect', 'Timestamp']]
data = data.dropna(axis=0)
data = data.reset_index(drop=True)

cons_list = pd.read_csv("data/constructs_input_test.csv")
cons = cons_list['ConstructId'].tolist()
skill2id = {}
for i in range(len(cons)):
    skill2id[cons[i]] = i

data.columns = ['user_id', 'skill_id', 'correct', 'time']
data = data[data['skill_id'].isin(cons)]
data = data.dropna(axis=0)
data = data.reset_index(drop=True)

_, idx, counts = np.unique(
    np.array(data["user_id"]), return_counts=True, return_index=True)
list_idx_to_remove_nest = [list(range(ind, ind + counts[counts < 10][c]))
                           for c, ind in enumerate(idx[counts < 10])]
list_idx_to_remove = [
    v for sublist in list_idx_to_remove_nest for v in sublist]  # flatten it
proc_data = data.drop(labels=list_idx_to_remove, axis=0).reset_index(drop=True)

proc_data['skill_id'] = proc_data['skill_id'].apply(lambda x: skill2id[x])
proc_data.to_csv("DKT/data/part_cons_train.csv")
