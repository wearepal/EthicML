"""
Shared Dataset for pytorch models
"""

from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data):
        self.features = np.array(data['x'].values, dtype=np.float32)
        self.class_labels = np.array(data['y'].values, dtype=np.float32)
        self.sens_labels = np.array(data['s'].values, dtype=np.float32)
        self.num = data['s'].count().values[0]
        self.s_size = data['s'].shape[1]
        self.y_size = data['y'].shape[1]
        self.size = data['x'].shape[1]
        self.x_names = data['x'].columns
        self.s_names = data['s'].columns
        self.y_names = data['y'].columns

    def __getitem__(self, index):
        return self.features[index], self.sens_labels[index], self.class_labels[index]

    def __len__(self):
        return self.num

    def names(self):
        return self.x_names, self.s_names, self.y_names
