import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import pandas as pd
import numpy as np

class ArgoverseDataset(Dataset):
    """Dataset class for Argoverse"""

    def __init__(self, data_path: str, transform=None):
        super(ArgoverseDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform

        self.csv_list = glob.glob(os.path.join(self.data_path, '*'))
        self.csv_list.sort()

    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        csv_path = self.csv_list[idx]
        with open(csv_path, 'rb') as f:
            data = pd.read_csv(f)

        if self.transform:
            data = self.transform(data)

        return data


def my_collate(batch):
    # extract 50 timestamps for X values and Y values for each scene in batch
    xs = [scene.loc[scene['OBJECT_TYPE'] == 'AGENT', 'X'] for scene in batch]
    ys = [scene.loc[scene['OBJECT_TYPE'] == 'AGENT', 'Y'] for scene in batch]

    # split X and Y values so that training has 45 datapoints and testing has 5 data points
    input_xs = [scene[:45] for scene in xs]
    input_ys = [scene[:45] for scene in ys]
    output_xs = [scene[45:] for scene in xs]
    output_ys = [scene[45:] for scene in ys]

    # Input: batch_size x 45 x 2
    inp = [np.dstack((x, y)).reshape(45, 2) for (x, y) in zip(input_xs, input_ys)]
    # Output: batch size x 5 x 2
    out = [np.dstack((x, y)).reshape(5, 2) for (x, y) in zip(output_xs, output_ys)]


    # Convert np.array into pytorch tensor
    inp = torch.FloatTensor(inp)
    out = torch.FloatTensor(out)

    return [inp, out]