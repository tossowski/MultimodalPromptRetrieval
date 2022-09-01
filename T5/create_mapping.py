import os


from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ROCO import ROCOFeatureDataset

MODEL_SAVE_FOLDER = "models"
MODEL_SAVE_PATH = "models/crossmodal_mapping.pt"

class CrossModalMapping(nn.Module):
    def __init__(self, in_dim = 512, out_dim = 512):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, batch):
        mapped_image_feats =  self.linear_relu_stack(batch["clip_image_features"])
        logit_scale = self.logit_scale.exp()
        logits_per_image_clip = logit_scale * batch["clip_text_features"] @ batch["clip_image_features"].t()
        logits_per_image_t5 = logit_scale * batch["t5_text_features"] @ mapped_image_feats.t()
        return logits_per_image_clip, logits_per_image_t5

    def loss(self, v1, v2):
        return torch.norm(v1 - v2, dim = 1)


dataset_train = ROCOFeatureDataset()
train_loader = DataLoader(dataset_train, 1, shuffle=True, num_workers=2)
model = CrossModalMapping(512, 512)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

best_valid_loss = float("inf")
best_epoch = 0
for epoch in range(100):
    #model.train()
    print(f"Starting epoch {epoch} ...")
    total = 0
    num_batch = 0
    for batch in tqdm(train_loader):
        clip_sim, t5_sim = model(batch)
        loss = model.loss(clip_sim, t5_sim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        num_batch += 1
    print(f"Average batch loss is {total / num_batch}")

    os.makedirs(MODEL_SAVE_FOLDER, exist_ok = True)
    torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, MODEL_SAVE_PATH)
    # print(f"Validation Loss: {valid_loss} | Best Validation Loss: {best_valid_loss} at epoch {best_epoch}")
    # if valid_loss < best_valid_loss:
    #     print(f"Saving model to {MODEL_SAVE_PATH} ...")
    #     os.makedirs(MODEL_SAVE_FOLDER, exist_ok = True)
    #     torch.save({'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict()}, MODEL_SAVE_PATH)
    #     best_valid_loss = valid_loss
    #     best_epoch = epoch
