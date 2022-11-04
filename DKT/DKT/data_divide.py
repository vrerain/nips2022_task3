import os
import pandas as pd
import numpy as np
import tqdm

user_list = []
student_list = []
path = os.path.join('data')
num_skill = 116


def parse_all_seq(students, data, questions):
    all_sequences = []
    for student_id in tqdm.tqdm(students, 'parse student sequence:\t'):
        if len(data[data.user_id == student_id]) < 40:
            continue
        student_list.append(student_id)
        student_sequence = parse_student_seq(
            data[data.user_id == student_id], questions)
        all_sequences.extend([student_sequence])
    return all_sequences


def parse_student_seq(student, questions):
    seq = student.sort_values('order_id')
    q = [questions[q] for q in seq.skill_id.tolist()]
    a = seq.correct.tolist()
    return q, a


def encode_onehot(sequences, max_step, num_questions):
    result = []

    index2 = 0
    for q, a in tqdm.tqdm(sequences, 'convert to one-hot format: '):
        length = len(q)
        # 提取学生序列
        for i in range(int(length/max_step) + (0 if length % max_step == 0 else 1)):
            user_list.append([student_list[index2]])
        # append questions' and answers' length to an integer multiple of max_step
        mod = 0 if length % max_step == 0 else (max_step - length % max_step)
        # 补成max_step的倍数
        onehot = np.zeros(shape=[length + mod, 2 * num_questions])
        for i, q_id in enumerate(q):
            index = int(q_id if a[i] > 0 else q_id + num_questions)
            onehot[i][index] = 1
        result = np.append(result, onehot)
        index2 += 1

    return result.reshape(-1, max_step, 2 * num_questions)


def divide_data(division_ratio):
    data = pd.read_csv(os.path.join(path, 'part_cons_train.csv'))
    data.columns = ['order_id', 'user_id',
                    'skill_id', 'correct', 'time']

    raw_question = data.skill_id.unique().tolist()

    # question id from 0 to (num_skill - 1)
    questions = {p: i for i, p in enumerate(raw_question)}

    # [(question_sequence_0, answer_sequence_0), ..., (question_sequence_n, answer_sequence_n)]
    sequences = parse_all_seq(data.user_id.unique(), data, questions)

    MAX_STEP = 300
    NUM_QUESTIONS = num_skill

    train_data = encode_onehot(sequences, MAX_STEP, NUM_QUESTIONS)
    if not os.path.exists(os.path.join(path, 'temp')):
        os.mkdir(os.path.join(path, 'temp'))
    rows = int(train_data.shape[0] * 0.85) + 1
    np.save(os.path.join(path, 'temp', 'train_data.npy'), train_data[:rows])
    np.save(os.path.join(path, 'temp', 'train_user_list.npy'),
            user_list[:rows])
    np.save(os.path.join(path, 'temp', 'test_data.npy'), train_data[rows:])
    np.save(os.path.join(path, 'temp', 'test_user_list.npy'), user_list[rows:])
    np.save(os.path.join(path, 'temp', 'all_data.npy'), train_data)
    np.save(os.path.join(path, 'temp', 'all_user_list.npy'), user_list)

    # return user_list
