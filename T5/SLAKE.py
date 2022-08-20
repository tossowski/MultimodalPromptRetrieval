import json
import numpy as np
from PIL import Image
import torch
import os
import pickle
import clip
from torch.utils.data import Dataset,DataLoader

def _load_dataset(dataroot, name):
    """Load entries

    img2id: dict {img -> id} id can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + '.json')
    
    samples_all = json.load(open(data_path))
    samples = [sample for sample in samples_all if sample['q_lang']=="en"]

    entries = []
    for entry in samples:
        sample = {'image_name' : entry['img_name'],
            'question': entry['question'],
            'answer' : entry['answer'],
            'task': entry['content_type'],
            'question_type': entry['answer_type'].lower()}
        entries.append(sample)

    
    return entries


class VQASLAKEFeatureDataset(Dataset):
    def __init__(self, name, dataroot):
        super(VQASLAKEFeatureDataset, self).__init__()
        self.name = name
        self.dataroot = dataroot
        self.entries = _load_dataset(dataroot, name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.preprocess = clip.load("ViT-B/32", device=device)
        
        images_path = os.path.join(dataroot, f'images_{name}.pkl')
        if os.path.exists(images_path):
            print(f"Loading existing images from {images_path}")
            with open(images_path, 'rb') as f:
                self.images = pickle.load(f)
            print(f"Loaded {len(self.images)} existing images")
        else:
            print(f"Creating images file: {images_path}")
            image_dict = {}
            for entry in self.entries:
                if entry['image_name'] in image_dict:
                    continue
                image_path = os.path.join(dataroot, "imgs", entry['image_name'])
                image = Image.open(image_path)
                image = self.preprocess(image)
                image_dict[entry['image_name']] = image
            with open(images_path, 'wb') as f:
                pickle.dump(image_dict, f)
            with open(images_path, 'rb') as f:
                self.images = pickle.load(f)
            print(f"Loaded {len(self.images)} existing images")

    def filter(self, qtype_list):
        self.entries = [x for x in self.entries if x["task"] in qtype_list]

    def __len__(self):
        return len(self.entries)
    

    def __getitem__(self, index):
        entry = self.entries[index]
        item = {}
        item["path_to_image"] = os.path.join(self.dataroot, "imgs", entry['image_name'])
        item['image'] = self.images[entry['image_name']]
        item['question'] = entry['question']
        item['answer'] = entry['answer']
        item['task'] = entry['task']
        item['question_type'] = entry['question_type']
        return item