from math import comb
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

import clip
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from T5VisionModel import T5VisionModel
from create_mapping import CrossModalMapping

class T5VisionModelFrozen(T5VisionModel):
    def __init__(self, vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128, use_image_info=True, vision_checkpoint=None, mapping_checkpoint=None):
        super().__init__(vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128, use_image_info=True, vision_checkpoint=None, mapping_checkpoint=None)
        self.T5_model.encoder.requires_grad_(False)
        for para in self.T5_model.parameters():
            print(para.requires_grad)
        
    

