from transformers import T5Tokenizer, T5ForConditionalGeneration

import clip
import torch
import torch.nn.functional as F
from torch import nn

class T5VisionModel(nn.Module):
    def __init__(self, vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_encoder = vision_encoder
        self.T5_version = T5_version
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.vision_model, _ = clip.load(self.vision_encoder, device=self.device)
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.T5_version)
        self.T5_model = T5ForConditionalGeneration.from_pretrained(self.T5_version)

        self.vision_model.visual = self.vision_model.visual.float()
        self.vision_model.visual.forward = self.get_image_token_features
        #print(self.T5_model.shared.embedding_dim)



    # Returns [batch_sz, grid ** 2, hidden_dim]
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

    # Pad attention masks with additional 1s so image embeddings aren't ignored
    def pad_attention_mask(self, masks, pad_amount):
        # Maybe parallelize later
        new_mask = []
        for i in range(masks.shape[0]):
            idx = sum(masks[i])
            new_mask.append(torch.cat([masks[i][:idx], torch.ones(pad_amount), masks[i][idx:]], 0))
        return torch.stack(new_mask, axis=0).to(self.device)

    def predict_sequence(self, batch):
        task_prefixes = [f"Answer the {x} question: " for x in batch['task']]
        image_prompts = [f"Based on the picture: " for x in batch['task']]
        
        encoding = self.tokenizer(
        [task_prefixes[i] + batch['question'][i] + image_prompts[i] for i in range(len(batch['question']))],
        padding="longest",
        max_length=self.max_source_length,
        truncation=True,
        return_tensors="pt",
        )

        image_embeddings = self.vision_model.visual(batch["image"].to(self.device))
        question_embedding = self.T5_model.shared(encoding["input_ids"].to(self.device))


        combined_embedding = torch.cat((question_embedding, image_embeddings), axis=1).to(self.device)
        #attention_mask = torch.cat((encoding.attention_mask), axis=1).to(self.device)
        attention_mask = self.pad_attention_mask(encoding.attention_mask, image_embeddings.shape[1])

        #print(model.get_input_embeddings())
        output_sequences = self.T5_model.generate(
        inputs_embeds = combined_embedding,
        attention_mask=attention_mask,
        do_sample=False,  # disable sampling to test if batching affects output
        max_new_tokens=20
        )

        predicted_answers = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        return predicted_answers


    def forward(self, batch):
        task_prefixes = [f"Answer the {x} question: " for x in batch['task']]
        image_prompts = [f"Based on the picture: " for x in batch['task']]
        
        image_embeddings = self.vision_model.encode_image(batch["image"].to(self.device))

        encoding = self.tokenizer(
        [task_prefixes[i] + batch['question'][i] + image_prompts[i] for i in range(len(batch['question']))],
        padding="longest",
        max_length=self.max_source_length,
        truncation=True,
        return_tensors="pt",
        )

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        question_embedding = self.T5_model.shared(input_ids.to(self.device))

        target_encoding = self.tokenizer(
        batch['answer'], padding="longest", max_length=self.max_target_length, truncation=True
        )

        labels = target_encoding.input_ids
        labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100


        # Adding vision embeddings to question
        #sprint(question_embedding.shape, image_embeddings.shape)
        #image_embeddings = torch.zeros_like(image_embeddings)
        combined_embedding = torch.cat((question_embedding, image_embeddings), axis=1)
        attention_mask = self.pad_attention_mask(encoding.attention_mask, image_embeddings.shape[1])
        
        combined_embedding = combined_embedding.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        loss = self.T5_model(inputs_embeds = combined_embedding, attention_mask=attention_mask, labels=labels).loss
        return loss
        

