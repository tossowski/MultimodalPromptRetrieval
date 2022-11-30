from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from SLAKE import VQASLAKEFeatureDataset
from matplotlib import pyplot as plt

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
from transformers import T5Tokenizer, T5ForConditionalGeneration
from create_mapping import CrossModalMapping

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

class_idx = json.load(open("imagenet/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


train_dataset = datasets.ImageFolder(
    "imagenet/val",
    _transform(224))

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
model.to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
T5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
mapping_model = CrossModalMapping(512, 512).to(device)
checkpoint = torch.load("models/crossmodal_mapping2.pt")
mapping_model.load_state_dict(checkpoint['model_state_dict'])

train_loader = DataLoader(train_dataset, 1, shuffle=True, num_workers=2)
image_vecs = []
text_vecs =  []
text_vecs_t5 = []
image_vecs_t5 = []
i = 0
for batch in tqdm(train_loader):
    text = f"A photo of a {idx2label[batch[1].item()]}"
    text = clip.tokenize([f"A photo of a {idx2label[batch[1].item()]}"]).to(device)

    image_features = model.encode_image(batch[0].to(device))
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    
    text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

    encoding = tokenizer(
        [f"A photo of a {idx2label[batch[1].item()]}"],
        padding="longest",
        max_length=512,
        truncation=True,
        return_tensors="pt",
        )
    caption_embedding = T5_model.shared(encoding["input_ids"].to(device))
    caption_embedding = torch.mean(caption_embedding, axis=1)
    caption_embedding = caption_embedding / caption_embedding.norm(dim=1, keepdim=True)

    # T5 Mapped image features
    t5_image_features = mapping_model.linear_relu_stack(text_features)
    #print(mapping_model.loss2(t5_image_features, caption_embedding))
    #t5_image_features = t5_image_features / t5_image_features.norm(dim=1, keepdim=True)


    image_features = image_features.detach().cpu().numpy()
    text_features = text_features.detach().cpu().numpy()
    caption_embedding = caption_embedding.detach().cpu().numpy()
    t5_image_features = t5_image_features.detach().cpu().numpy()

    image_vecs.append(image_features[0])
    text_vecs.append(text_features[0])
    text_vecs_t5.append(caption_embedding[0])
    image_vecs_t5.append(t5_image_features[0])
    i += 1
    if i == 2000:
        break


image_data = np.stack(image_vecs, axis=0)
text_data = np.stack(text_vecs, axis=0)
t5_data = np.stack(text_vecs_t5)
t5_images = np.stack(image_vecs_t5)
data = np.concatenate((image_data, text_data, t5_data, t5_images), axis=0)
num_images = len(image_data)

scaler = StandardScaler()
scaler.fit(data)
data=scaler.transform(data)    


pca = PCA()
fitted_data = pca.fit_transform(data)
x_new_image = fitted_data[:len(image_data)]
x_new_text = fitted_data[len(image_data): 2 * len(image_data)]
x_new_t5_text = fitted_data[2 * len(image_data):3 * len(image_data)]
x_new_t5_images = fitted_data[3 * len(image_data):]

# np.random.seed(1)
# x_new = TSNE(n_components=2).fit_transform(data)
print(len(x_new_image), len(x_new_text), len(x_new_t5_text), len(x_new_t5_images))

#print(pca.explained_variance_ratio_)

fig = plt.figure()
scatter_images = plt.scatter(x_new_image[:,0], x_new_image[:,1], label="image_features")
scatter_text = plt.scatter(x_new_text[:,0], x_new_text[:,1], label="text_features")
scatter_T5 = plt.scatter(x_new_t5_text[:,0], x_new_t5_text[:,1], label="t5_text_features")
scatter_T5_image = plt.scatter(x_new_t5_images[:,0], x_new_t5_images[:,1], label="t5_image_features")


plt.title("CLIP and T5 Image and Text Features on Imagenet Data")
names = ['Image Features', 'Caption Features']
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Feature Type")

# for i in range(len(x_new_image) // 2):
#     if i % 100 == 0:
#         x1, y1 = [x_new_image[i][0], x_new_text[i][0]], [x_new_image[i][1], x_new_text[i][1]]

#         plt.plot(x1, y1, marker = 'o')

plt.savefig(f"clip_imagenet.png")
