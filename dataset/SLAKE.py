from cmath import inf
import json
import numpy as np
from PIL import Image
import torch
import os
import pickle
import clip
from torch.utils.data import Dataset,DataLoader
from dataset.VQAFeatureDataset import VQADataset

class VQASLAKEFeatureDataset(VQADataset):
    def __init__(self, name, dataroot, device = "cuda" if torch.cuda.is_available() else "cpu"):
        super(VQASLAKEFeatureDataset, self).__init__(name, dataroot, device)