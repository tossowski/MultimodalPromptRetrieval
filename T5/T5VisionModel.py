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

class T5VisionModel(nn.Module):
    def __init__(self, vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128, use_image_info=True, vision_checkpoint=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_encoder = vision_encoder
        self.T5_version = T5_version
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.use_image_info = use_image_info
        self.vision_model, _ = clip.load(self.vision_encoder, device=self.device)
        
        if vision_checkpoint:
            print(f"Loading pretrained vision checkpoint: {vision_checkpoint}")
            checkpoint = torch.load(vision_checkpoint)
            self.vision_model.load_state_dict(checkpoint['state_dict'])

        self.vision_model = self.vision_model.float()
        self.vision_model.visual.forward = self.get_image_token_features

        self.tokenizer = T5Tokenizer.from_pretrained(self.T5_version)
        self.tokenizer.add_tokens(["[itk]"])
        self.T5_model = T5ForConditionalGeneration.from_pretrained(self.T5_version)
        self.T5_model.resize_token_embeddings(len(self.tokenizer))

        self.load_clip_to_t5_mapping()

    def load_clip_to_t5_mapping(self):
        os.makedirs("mapping", exist_ok=True)
        PATH_TO_MAPPING = f"mapping/{self.vision_encoder.replace('/','_')}_{self.T5_version}.npy"
        if os.path.exists(PATH_TO_MAPPING):
            print(f"Loading mapping from {PATH_TO_MAPPING}")
            self.W = torch.FloatTensor(np.load(PATH_TO_MAPPING)).to(self.device)
            self.W /= 20
        else:
            vocab = self.tokenizer.get_vocab()
            A = []
            B = []
            print(f"Creating W from {self.vision_encoder} to {self.T5_version} ...")
            for word, vocab_id in tqdm(vocab.items()):
                clip_word_encoding = clip.tokenize([word]).to(self.device)
                clip_word_emb = self.vision_model.encode_text(clip_word_encoding)
                T5_word_emb = self.T5_model.shared(torch.LongTensor([vocab_id]))
                A.append(clip_word_emb.detach().cpu().numpy())
                B.append(T5_word_emb.detach().cpu().numpy())
            A = np.concatenate(A, axis=0)
            B = np.concatenate(B, axis=0)

            # Least squares solution
            W = np.linalg.inv(A.T @ A) @ A.T @ B
            print(f"Saved W to {PATH_TO_MAPPING}")
            np.save(PATH_TO_MAPPING, W)
            self.W = torch.FloatTensor(W).to(self.device)


    # Returns [batch_sz, grid ** 2 + 1, hidden_dim]
    def get_image_token_features(self, x):
        x = self.vision_model.visual.conv1(x)  # shape = [*, width, grid, grid]
        
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.vision_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.vision_model.visual.positional_embedding.to(x.dtype)
        x = self.vision_model.visual.ln_pre(x)
        

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vision_model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        
        x = self.vision_model.visual.ln_post(x)

        if self.vision_model.visual.proj is not None:
            x = x @ self.vision_model.visual.proj

        #x = x @ self.W

        return x

    def prepare_input(self, batch):
        task_prefixes = [f"Answer the {x} question: " for x in batch['task']]
        image_prompts = [" Based on the picture: " for x in batch['task']]

        image_embeddings = self.vision_model.visual(batch["image"].to(self.device))

        image_tokens = "[itk] " * image_embeddings.shape[1]
        image_tokens = image_tokens[:-1]
    
        if self.use_image_info:
            input_sentences = [task_prefixes[i] + batch['question'][i] + image_prompts[i] + image_tokens for i in range(len(batch['question']))]
        else:
            input_sentences = [task_prefixes[i] + batch['question'][i] for i in range(len(batch['question']))]
        
        encoding = self.tokenizer(
        input_sentences,
        padding="longest",
        max_length=self.max_source_length,
        truncation=True,
        return_tensors="pt",
        )

        question_embedding = self.T5_model.shared(encoding["input_ids"].to(self.device))
        
        
        
        attention_mask = encoding.attention_mask.to(self.device)
        if self.use_image_info:
            combined_embedding = self.insert_image_features(image_embeddings, question_embedding, encoding.attention_mask)
        else:
            combined_embedding = question_embedding.to(self.device)
        
        norm = combined_embedding.pow(2).sum(keepdim=True, dim=2).sqrt()
        combined_embedding = combined_embedding / norm

        # PCA CODE

        data = combined_embedding.detach().cpu().numpy()[0]

        scaler = StandardScaler()
        scaler.fit(data)
        data=scaler.transform(data)    
        labels = np.zeros(data.shape[0])
        labels[-51:-1] = 1

        pca = PCA()
        x_new = pca.fit_transform(data)
        my_colors = np.where(labels == 1, "red", "blue")
        fig = plt.figure()
        plt.scatter(x_new[:,0], x_new[:,1],color=my_colors)
        plt.savefig(f"pca_{batch['question_id'][0]}.png")

        test = combined_embedding.detach().cpu().numpy()
        fig, ax = plt.subplots(1, len(test[0]), figsize=(200,10))
        for i in range(len(test[0])):
            ax[i].hist(test[0][i])
        plt.savefig("test.png")
        
        return combined_embedding, attention_mask, encoding

    # Pad attention masks with additional 1s so image features aren't ignored
    def insert_image_features(self, image_features, question_embedding, attention_masks):
        # Maybe parallelize later
        n_image_tokens = image_features.shape[1]
        len_sentence = question_embedding.shape[1]
        for i in range(image_features.shape[0]):
            n_padding = len_sentence - sum(attention_masks[i])
            question_embedding[i, len_sentence - n_padding - n_image_tokens - 1:len_sentence - n_padding - 1, :] = image_features[i]
        return question_embedding

    def predict_sequence(self, batch):

        combined_embedding, attention_mask, _ = self.prepare_input(batch)

        output_sequences = self.T5_model.generate(
        inputs_embeds = combined_embedding,
        attention_mask=attention_mask,
        do_sample=False,  # disable sampling to test if batching affects output
        max_new_tokens=20
        )

        predicted_answers = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        return predicted_answers


    def forward(self, batch):

        combined_embedding, attention_mask, _ = self.prepare_input(batch)
        
        target_encoding = self.tokenizer(
        batch['answer'], padding="longest", max_length=self.max_target_length, truncation=True
        )

        labels = target_encoding.input_ids
        labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels = labels.to(self.device)


        loss = self.T5_model(inputs_embeds = combined_embedding, attention_mask=attention_mask, labels=labels).loss
        return loss
        

