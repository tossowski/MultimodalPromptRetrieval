from transformers import T5Tokenizer, T5ForConditionalGeneration

import clip
import torch
import torch.nn.functional as F
from torch import nn

class T5VisionModel(nn.Module):
    def __init__(self, vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128, use_image_info=True):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_encoder = vision_encoder
        self.T5_version = T5_version
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.use_image_info = use_image_info
        self.vision_model, _ = clip.load(self.vision_encoder, device=self.device)
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.T5_version)
        self.tokenizer.add_tokens(["[itk]"])
        self.T5_model = T5ForConditionalGeneration.from_pretrained(self.T5_version)
        self.T5_model.resize_token_embeddings(len(self.tokenizer))

        self.vision_model.visual = self.vision_model.visual.float()
        self.vision_model.visual.forward = self.get_image_token_features
        #print(self.T5_model.shared.embedding_dim)



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
        

