import torch
import os, glob
import random

import torch, sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
            filename = 'data/names_train.csv.gz' if is_train_set else 'data/names_test.csv.gz'
            with gzip.open(filename, 'rt') as f:
                reader = csv.reader(f)
                rows = list(reader)
            self.names = [row[0] for row in rows]
            self.len = len(self.names)
            self.countries = [row[1] for row in rows]
            self.country_list = list(sorted(set(self.countries)))
            self.country_dict = self.getCountryDict()
            self.country_num = len(self.country_list)
        def __getitem__(self, index):
            return self.names[index], self.country_dict[self.countries[index]]
        def __len__(self):
            return self.len