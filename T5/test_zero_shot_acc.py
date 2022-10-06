from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from SLAKE import VQASLAKEFeatureDataset
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import torch
import clip
import json
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
from transformers import CLIPTokenizer
from torch import nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
class_idx = json.load(open("imagenet/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
possible_captions = [f"A photo of a {x}" for x in idx2label]


train_dataset = datasets.ImageFolder(
    "imagenet/val",
    _transform(224))

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.to(device)

all_text_features = []
for caption in possible_captions:
    text = clip.tokenize([caption]).to(device)
    text_feats = model.encode_text(text)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    all_text_features.append(text_feats)
#print(all_text_features[0].shape)
text_features = torch.cat(all_text_features, dim=0)
train_loader = DataLoader(train_dataset, 1, shuffle=True, num_workers=2)
total = 0
correct = 0
for batch in tqdm(train_loader):
    with torch.no_grad():
        image_features = model.encode_image(batch[0].to(device))
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        # Print the result
        if batch[1].item() in indices:
            correct += 1
        total += 1
        #print(indices)
print(correct/total) 
    
