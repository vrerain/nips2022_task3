import torch
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
import numpy as np
import os

path = os.path.join('data')


class subDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.Data[index])
        label = torch.IntTensor(self.Label[index])
        return data, label


class subDataset2(Dataset):
    def __init__(self, Data):
        self.Data = Data

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.Data[index])
        return data


def get_data_loader(data_path, data_path2, batch_size, shuffle=False):
    data = np.load(data_path)
    data2 = np.load(data_path2)
    dataset = subDataset(data, data2)
    data_loader = Data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def get_data_loader2(data_path, batch_size, shuffle=False):
    data = np.load(data_path)
    dataset = subDataset2(data)
    return Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class TrainDataLoader(object):
    def __init__(self, batch_size):
        train_loader = get_data_loader(os.path.join(path, 'temp', 'train_data.npy'), os.path.join(
            path, 'temp', 'train_user_list.npy'), batch_size, True)
        self.train_loader = train_loader


class ValTestDataLoader(object):
    def __init__(self, batch_size,  d_type='predict'):
        test_loader = get_data_loader2(os.path.join(path,
                                                    'temp', 'test_data.npy'), batch_size, False)
        self.test_loader = test_loader


class GeneratroDataLoader(object):
    def __init__(self, batch_size):
        gen_loader = get_data_loader(os.path.join(path, 'temp', 'all_data.npy'), os.path.join(
            path, 'temp', 'all_user_list.npy'), batch_size, True)
        self.gen_loader = gen_loader
