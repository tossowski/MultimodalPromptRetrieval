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

class T5VisionModelPredictionHead(T5VisionModel):
    def __init__(self, device, num_classes, vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128, use_image_info=True, vision_checkpoint=None, mapping_checkpoint=None, retrieval_function=None, use_quantifier=True):
        super().__init__(device, vision_encoder = vision_encoder, T5_version = T5_version, max_source_length = max_source_length, max_target_length = max_target_length, use_image_info=use_image_info, vision_checkpoint=vision_checkpoint, mapping_checkpoint=mapping_checkpoint, retrieval_function=retrieval_function, use_quantifier=use_quantifier)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
 
        self.num_classes = num_classes

        self.prediction_head = nn.Linear(512, self.num_classes)
        self.prediction_dropout = nn.Dropout(0.1) 

    def predict(self, batch):

        combined_embedding, attention_mask, _ = self.prepare_input(batch)
        
        target_encoding = self.tokenizer(
        batch['answer'], padding="longest", max_length=self.max_target_length, truncation=True
        )

        labels = target_encoding.input_ids
        labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels = labels.to(self.device)

        
        #print(batch['labels'])
        output = self.T5_model(inputs_embeds = combined_embedding, attention_mask=attention_mask, labels=labels)
        #print(output['encoder_last_hidden_state'].shape)
        last_hidden_state = output['encoder_last_hidden_state'][:,-1,:]
        dropped = self.prediction_dropout(last_hidden_state)
        logits = self.prediction_head(dropped)
        predictions = torch.argmax(logits, dim = 1)
        #probs = logits.softmax(dim = 1)
        #print(batch['labels'])
        return predictions


    def forward(self, batch):

        combined_embedding, attention_mask, _ = self.prepare_input(batch)
        
        target_encoding = self.tokenizer(
        batch['answer'], padding="longest", max_length=self.max_target_length, truncation=True
        )

        labels = target_encoding.input_ids
        labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels = labels.to(self.device)

        
        #print(batch['labels'])
        output = self.T5_model(inputs_embeds = combined_embedding, attention_mask=attention_mask, labels=labels)
        #print(output['encoder_last_hidden_state'].shape)
        last_hidden_state = output['encoder_last_hidden_state'][:,-1,:]
        dropped = self.prediction_dropout(last_hidden_state)
        logits = self.prediction_head(dropped)
        #probs = logits.softmax(dim = 1)
        #print(batch['labels'])
        return self.loss_fn(logits, batch['label'].to(self.device))
        

