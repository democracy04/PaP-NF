import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, data_path='ETTm2.csv', flag='train', 
                 size=None, features='M', target='OT', scale=True):
    
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        
        # Base Prompt
        self.prompt_text = (
            "The Electricity Transformer Temperature (ETT) dataset contains "
            "time series data from electricity transformers, including load "
            "and oil temperature indicators. The task is to forecast future "
            "transformer temperatures based on historical patterns."
        )

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))


        border1s = [0, 12*30*24 - self.seq_len, 12*30*24 + 4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] 
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        x_mean = np.mean(seq_x)
        x_std = np.std(seq_x)
        x_min = np.min(seq_x)
        x_max = np.max(seq_x)
        
        dynamic_prompt = (
            f"{self.prompt_text} "
            f"Input statistics: Mean {x_mean:.4f}, Std {x_std:.4f}, "
            f"Min {x_min:.4f}, Max {x_max:.4f}. "
            "Predict the future values."
        )

        return (
            torch.tensor(seq_x, dtype=torch.float32), 
            torch.tensor(seq_y, dtype=torch.float32),
            dynamic_prompt 
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
