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

class VisionTransformerTokens(nn.Module):
    def __init__(self, vision_transformer):
        super().__init__()
        self.visual = vision_transformer
        self.visual.forward = self.get_image_token_features

    def get_image_token_features(self, x: torch.Tensor):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x)

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        return x

class_idx = json.load(open("imagenet/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]


train_dataset = datasets.ImageFolder(
    "imagenet/val",
    _transform(224))

images = train_dataset.imgs
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image_encoder = VisionTransformerTokens(model.visual)
model.to(device)
image_encoder.to(device)
tokenizer = _Tokenizer()

train_loader = DataLoader(train_dataset, 1, shuffle=True, num_workers=2)
for num, batch in tqdm(enumerate(images)):
    text = f"A photo of a {idx2label[batch[1]]}"
    text = clip.tokenize([f"A photo of a {idx2label[batch[1]]}"]).to(device)
    
    image = _transform(224)(Image.open(batch[0]))
    torch_image = torch.unsqueeze(image, 0)
    zero_idx = 0
    for i in range(len(text[0])):
        if text[0][i] != 0:
            zero_idx += 1
        else:
            break


    image_features = model.encode_image(torch_image.to(device)).detach().cpu().numpy()[0]
    #image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #print(image_features.shape)
    x = model.token_embedding(text).type(model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x).type(model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    text_features = (x @ model.text_projection)[0,:zero_idx,:].detach().cpu().numpy()
   # print(image_features.shape, text_features.shape)
    
    similarity_scores = text_features @ image_features.T
    softmax_scores = torch.nn.functional.softmax(torch.FloatTensor(similarity_scores), dim = 1)
    minimum = torch.min(torch.FloatTensor(similarity_scores), 1, keepdim=True)[0]
    maximum = torch.max(torch.FloatTensor(similarity_scores), 1, keepdim=True)[0]
   # print(similarity_scores.shape, minimum.shape)
    minmaxxed = (torch.FloatTensor(similarity_scores) - minimum) / (maximum - minimum)
 
    most_relevant_token = np.argmax(similarity_scores, axis = 1)
    caption = tokenizer.decode(text[0].detach().cpu().numpy()[:zero_idx])
    caption = ['<s>', 'a'] + caption.split()[1:-1]

    image = Image.open(batch[0])
    image_x_ticks = np.linspace(0, image.width, 7 + 1)
    image_y_ticks = np.linspace(0, image.height, 7 + 1)
    grid_x_length = image_x_ticks[1] - image_x_ticks[0]
    grid_y_length = image_y_ticks[1] - image_y_ticks[0]

    for i, word in enumerate(caption):
        token_num = most_relevant_token[i] - 1
        if word in ["<s>", "a", "photo", "of"]:
            continue
        plt.imshow(image)
        plt.xticks(image_x_ticks)
        plt.yticks(image_y_ticks)
        plt.grid()
        ax = plt.gca()
        alphas = minmaxxed[i].detach().cpu().numpy()[1:]
        for l in range(7):
            for m in range(7):

                rect = patches.Rectangle((image_x_ticks[m], image_y_ticks[l]), grid_x_length, grid_y_length, linewidth=1, fill=True, facecolor="red", alpha=alphas[7 * l + m])
                ax.add_patch(rect)

        os.makedirs(os.path.join("CLIPTOKEN", word), exist_ok=True)
        plt.savefig(os.path.join("CLIPTOKEN", word, f"{token_num}.png"))

        plt.close()

        
    #text_features = text_features / text_features.norm(dim=1, keepdim=True)
    

    

    