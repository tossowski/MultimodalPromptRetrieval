import pandas as pd
import torch
import os
from VQAFeatureDataset import VQADataset

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




class VQARADFeatureDataset(VQADataset):
    def __init__(self, name, dataroot, device = "cuda" if torch.cuda.is_available() else "cpu"):
        super(VQARADFeatureDataset, self).__init__(name , dataroot, device)
        

    def _load_dataset(sself, dataroot, name):

        data_path = os.path.join(dataroot, f'{name}.json')
        samples_all = pd.read_json(data_path)

        entries = []
        for idx, entry in samples_all.iterrows():
            for qtype in entry["question_type"].split(", "):

                sample = {'image_name' : entry['image_name'],
                    'question_id': str(entry['qid']),
                    'question': entry['question'].lower(),
                    'answer' : str(entry['answer']).lower(),
                    'task': qtype_map[qtype],
                    'question_type': entry['answer_type'].lower()}
                entries.append(sample)

        
        return entries