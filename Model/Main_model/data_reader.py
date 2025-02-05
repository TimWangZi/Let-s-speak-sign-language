import json
import torch
from torch.utils.data import DataLoader, Dataset

class TrainDataLoader(DataLoader):
    def __init__(self, data):
        None
    def __len__(self):
        return super().__len__()
    def __getitem__(self ,idx):
        None
