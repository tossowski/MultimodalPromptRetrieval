import pandas as pd
import numpy as np
from PIL import Image
import torch
import os
import pickle
import clip
from torch.utils.data import Dataset,DataLoader

qtype_map = {
    "PRES": "Presence",
    "ABN": "Abnormality",
    "MODALITY": "Modality",
    "ORGAN": "Organ",
    "PLANE": "Plane",
    "OTHER": "Other",
    "SIZE": "Size",
    "ATTRIB": "Attribute",
    "COLOR": "Color",
    "ATRIB": "Attribute",
    "PRSE": "Presence",
    "POS": "Position",
    "COUNT": "Quantity",
    "Other": "Other"
}

def _load_dataset(dataroot, name):
    """Load entries

    img2id: dict {img -> id} id can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, 'VQA_RAD.xlsx')
    samples_all = pd.read_excel(data_path)

    if name == "test":
        samples_all = samples_all[samples_all["QID_para"].str.contains(name)]
    else:
        samples_all = samples_all[~samples_all["QID_para"].str.contains(name)]

    entries = []
    for idx, entry in samples_all.iterrows():
        for qtype in entry["Q_TYPE"].split(", "):

            sample = {'image_name' : entry['IMAGEID'].split("/")[-1],
                'question': entry['QUESTION'],
                'answer' : str(entry['ANSWER']),
                'task': qtype_map[qtype],
                'question_type': entry['A_TYPE'].lower()}
            entries.append(sample)

    
    return entries


class VQARADFeatureDataset(Dataset):
    def __init__(self, name, dataroot):
        super(VQARADFeatureDataset, self).__init__()
        self.name = name
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

    def __len__(self):
        return len(self.entries)
    

    def __getitem__(self, index):
        entry = self.entries[index]
        item = {}
        item['image'] = self.images[entry['image_name']]
        item['question'] = entry['question']
        item['answer'] = entry['answer']
        item['task'] = entry['task']
        item['question_type'] = entry['question_type']

        return item