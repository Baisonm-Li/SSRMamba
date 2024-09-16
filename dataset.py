import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NPZDataset(torch.utils.data.Dataset):
    def __init__(self,data_file):
        super(NPZDataset, self).__init__()
        self.npz_data = np.load(data_file)
        self.rgb_list = self.npz_data['rgb']
        self.hsi_list = self.npz_data['hsi']
        self.l = len(self.rgb_list)
    def __getitem__(self, index):
        return torch.from_numpy(self.rgb_list[index]).float(), torch.from_numpy(self.hsi_list[index]).float()

    def __len__(self):
        return self.l