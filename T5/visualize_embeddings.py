from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from SLAKE import VQASLAKEFeatureDataset
from matplotlib import pyplot as plt

import torch
import clip
import json
import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE

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

# class_idx = json.load(open("imagenet/imagenet_class_index.json"))
# idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


# train_dataset = datasets.ImageFolder(
#     "imagenet/val",
#     _transform(224))

CFG = json.load(open("config/config.json"))

data_name = CFG["dataset"]

train_dataset = VQASLAKEFeatureDataset("train", os.path.join(CFG["datafolder"],data_name))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.to(device)


train_loader = DataLoader(train_dataset, 1, shuffle=True, num_workers=2)
image_vecs = []
text_vecs =  []
for i, batch in enumerate(train_loader):
    #text = f"A photo of a {idx2label[batch[1].item()]}"
    #text = clip.tokenize([f"A photo of a {idx2label[batch[1].item()]}"]).to(device)
    text = clip.tokenize(batch["question"]).to(device)

    #image_features = model.encode_image(batch[0].to(device))
    image_features = model.encode_image(batch["image"].to(device))
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    image_features = image_features.detach().cpu().numpy()
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    text_features = text_features.detach().cpu().numpy()


    image_vecs.append(image_features[0])
    text_vecs.append(text_features[0])
    if i == 2000:
        break

image_data = np.stack(image_vecs, axis=0)
text_data = np.stack(text_vecs, axis=0)
data = np.concatenate((image_data, text_data), axis=0)
num_images = len(image_data)

scaler = StandardScaler()
scaler.fit(data)
data=scaler.transform(data)    


pca = PCA()
x_new = pca.fit_transform(data)
#np.random.seed(1)
#x_new = TSNE(n_components=2).fit_transform(data)


print(pca.explained_variance_ratio_)

fig = plt.figure()
scatter_images = plt.scatter(x_new[:len(image_data),0], x_new[:len(image_data),1], label="image_features")
scatter_text = plt.scatter(x_new[len(image_data):,0], x_new[len(image_data):,1], label="text_features")

plt.title("CLIP Image and Text Features on Medical Data")
names = ['Image Features', 'Question Features']
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Feature Type")

for i in range(len(x_new) // 2):
    if i % 100 == 0:
        x1, y1 = [x_new[i][0], x_new[i + len(image_data)][0]], [x_new[i][1], x_new[i + len(image_data)][1]]

        plt.plot(x1, y1, marker = 'o')

plt.savefig(f"clip_medical.png")
