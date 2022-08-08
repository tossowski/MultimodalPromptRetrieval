import json
import numpy as np
from PIL import Image
import torch
import os
import pickle
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
            'task': entry['content_type']}
        entries.append(sample)

    
    return entries


class VQASLAKEFeatureDataset(Dataset):
    def __init__(self, name, dataroot):
        super(VQASLAKEFeatureDataset, self).__init__()
        self.name = name
        self.entries = _load_dataset(dataroot, name)
        
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
                image = image.resize((224, 224), Image.ANTIALIAS)
                np_image = np.array(image)
                image_dict[entry['image_name']] = np_image
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
        return item