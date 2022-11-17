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
        PATH_TO_CACHED_FEATURES = f"/data/ossowski/roco-dataset/data/{clip_type}/{mode}"
        os.makedirs(PATH_TO_CACHED_FEATURES, exist_ok=True)

        if os.path.exists(os.path.join(f"{PATH_TO_CACHED_FEATURES}", "clip_text.npy")):

            self.clip_text_features = np.load(os.path.join(f"{PATH_TO_CACHED_FEATURES}", "clip_text.npy"))
            self.clip_image_features = np.load(os.path.join(f"{PATH_TO_CACHED_FEATURES}", "clip_images.npy"))
            self.t5_text_features = np.load(os.path.join(f"{PATH_TO_CACHED_FEATURES}", "T5_text.npy"))
        else:
            print("Creating Features from ROCO Dataset ...")
            self.create_features(PATH_TO_CACHED_FEATURES)

    def create_features(self, path):
        PATH_TO_CACHED_FEATURES = path


        model, preprocess = clip.load("ViT-B/32", device=self.device)

        if self.clip_type == "PubMedClip":
            checkpoint = torch.load("models/PubMedCLIP_ViT32.pth", map_location=torch.device(self.device))
            model.load_state_dict(checkpoint['state_dict'])
        model = model.float()
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        T5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(self.device)

        T5_text_feats = []
        clip_text_feats = []
        clip_image_feats = []

        PATH_TO_DATA = os.path.join(f"/data/ossowski/roco-dataset/data/{self.mode}", "radiology")
        with open(os.path.join(PATH_TO_DATA, "captions.txt"), "r") as f:
            num_lines = sum(1 for line in open(os.path.join(PATH_TO_DATA, "captions.txt"),'r'))
            for _, line in enumerate(tqdm(f, total = num_lines)):
                image_id, caption = line.split("\t")
                try:
                    image = preprocess(Image.open(os.path.join(PATH_TO_DATA, "images", f"{image_id}.jpg"))).unsqueeze(0).to(device)
                except:
                    print(os.path.join(PATH_TO_DATA, "images", f"{image_id}.jpg") + " not found!")
                    continue
                text = clip.tokenize([caption], truncate=True).to(self.device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    encoding = tokenizer(
                    [caption],
                    padding="longest",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                    )
                    caption_embedding = T5_model.shared(encoding["input_ids"].to(device))
                    caption_embedding = torch.mean(caption_embedding, axis=1)
                    caption_embedding = caption_embedding / caption_embedding.norm(dim=1, keepdim=True)
                    clip_image_feats.append(image_features.detach().cpu().numpy())
                    clip_text_feats.append(text_features.detach().cpu().numpy())

                    T5_text_feats.append(caption_embedding.detach().cpu().numpy())

        final_image_feats = np.concatenate(clip_image_feats, axis=0)
        final_text_feats = np.concatenate(clip_text_feats, axis=0)

        final_T5_text_feats = np.concatenate(T5_text_feats, axis=0)

        np.save(os.path.join(f"{PATH_TO_CACHED_FEATURES}", "clip_text.npy"), final_text_feats)
        np.save(os.path.join(f"{PATH_TO_CACHED_FEATURES}", "clip_images.npy"), final_image_feats)
        np.save(os.path.join(f"{PATH_TO_CACHED_FEATURES}", "T5_text.npy"), final_T5_text_feats)

        self.clip_text_features = np.load(os.path.join(f"{PATH_TO_CACHED_FEATURES}", "clip_text.npy"))
        self.clip_image_features = np.load(os.path.join(f"{PATH_TO_CACHED_FEATURES}", "clip_images.npy"))
        self.t5_text_features = np.load(os.path.join(f"{PATH_TO_CACHED_FEATURES}", "T5_text.npy"))
    
    # def __len__(self):
    #     return len(self.entries)
    

    # def __getitem__(self, index):
    #     item = {}
    #     entry = self.entries[index]

    #     item['image'] = self.images[entry['image_name']]
    #     item['question'] = entry['question']
    #     item['answer'] = entry['answer']
    #     item['task'] = entry['task']
    #     item['question_type'] = entry['question_type']
    #     # item["clip_text_features"] = self.clip_image_features[index]
    #     # item['clip_image_features'] = self.clip_text_features[index]
    #     # item['t5_text_features'] = self.t5_text_features[index]

    #     return item