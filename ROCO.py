from cmath import inf
import json
import numpy as np
from PIL import Image
import torch
import os
import pandas as pd
import clip
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from VQAFeatureDataset import VQADataset

class ROCOFeatureDataset(VQADataset):

    def _load_dataset(self, dataroot, name):

        data_path = os.path.join(dataroot, f'{name}.csv')
        samples_all = pd.read_csv(data_path)
        entries = []
        for idx, entry in samples_all.iterrows():
            
            sample = {'image_name' : entry['image_id'],
                'question': entry['question'].lower(),
                'answer' : str(entry['answer']).lower(),
                'task': entry['q_type'],
                'question_id': idx,
                'question_type': entry['question_type'].lower()}
            entries.append(sample)

        
        return entries

    def __init__(self, name, dataroot, mode="train", clip_type="PubMedClip", device = "cuda"):
        super(ROCOFeatureDataset, self).__init__(name, dataroot, device)

        self.clip_type = clip_type
        self.device = device
        self.mode = mode

   