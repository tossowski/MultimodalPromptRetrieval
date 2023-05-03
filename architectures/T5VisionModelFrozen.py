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
from architectures.T5VisionModel import T5VisionModel
from create_mapping import CrossModalMapping

class T5VisionModelFrozen(T5VisionModel):
    def __init__(self, vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128, use_image_info=True, vision_checkpoint=None, mapping_checkpoint=None, retrieval_function=None):
        super().__init__(vision_encoder =vision_encoder, T5_version = T5_version, max_source_length = max_source_length, max_target_length = max_target_length, use_image_info=use_image_info, vision_checkpoint=vision_checkpoint, mapping_checkpoint=mapping_checkpoint, retrieval_function=retrieval_function)

        self.T5_model.encoder.requires_grad_(False)
        self.T5_model.decoder.requires_grad_(False)
        self.vision_model.requires_grad_(False)
        self.T5_model.shared.requires_grad_(True)
        trainable_params = 0
        
        for para in self.T5_model.parameters():
            if para.requires_grad:
                trainable_params += np.prod(para.size())

        print(f"Freezing T5 model to have {trainable_params} trainable parameters ...")
    

