from cmath import inf
import json
import numpy as np
from PIL import Image
import torch
import os
import pickle
import clip
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration


class ROCOFeatureDataset(Dataset):
    def __init__(self, mode="train", clip_type="PubMedClip"):
        super(ROCOFeatureDataset, self).__init__()

        self.clip_type = clip_type
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

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, preprocess = clip.load("ViT-B/32", device=device)

        if self.clip_type == "PubMedClip":
            checkpoint = torch.load("models/PubMedCLIP_ViT32.pth")
            model.load_state_dict(checkpoint['state_dict'])
        model = model.float()
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        T5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

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
                text = clip.tokenize([caption], truncate=True).to(device)
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
    
    def __len__(self):
        return len(self.clip_text_features)
    

    def __getitem__(self, index):
        item = {}
        item["clip_text_features"] = self.clip_image_features[index]
        item['clip_image_features'] = self.clip_text_features[index]
        item['t5_text_features'] = self.t5_text_features[index]

        return item