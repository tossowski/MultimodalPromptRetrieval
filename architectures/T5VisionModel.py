from math import sqrt
from transformers import T5Tokenizer, T5ForConditionalGeneration
import clip
import torch
from torch import nn
from create_mapping import CrossModalMapping
from utils import cosine_similarity
from torch.autograd import Variable
import numpy as np


class T5VisionModel(nn.Module):
    def __init__(self, vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128, use_image_info=True, vision_checkpoint=None, mapping_checkpoint=None, retrieval_function=None, use_quantifier=True):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #self.device="cpu"

        self.vision_encoder = vision_encoder
        self.T5_version = T5_version
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.use_image_info = use_image_info
        self.retrieval_function = retrieval_function
        self.use_quantifier = use_quantifier
        self.use_mapping = bool(mapping_checkpoint)
        self.vision_model, _ = clip.load(self.vision_encoder, device=self.device)
        self.map_to_large = False

        for p in self.vision_model.parameters():
            p.requires_grad = False

        if mapping_checkpoint:
            #print(self.T5_version.shared)
            print(f"Loading Mapping Model: {mapping_checkpoint}")
            self.mapping = CrossModalMapping(512, 512).to(self.device)
            checkpoint = torch.load(mapping_checkpoint, map_location=torch.device(self.device))
            self.mapping.load_state_dict(checkpoint['model_state_dict'])
        
        if vision_checkpoint:
            print(f"Loading pretrained vision checkpoint: {vision_checkpoint}")
            checkpoint = torch.load(vision_checkpoint, map_location=torch.device(self.device))
            self.vision_model.load_state_dict(checkpoint['state_dict'])

        self.vision_model = self.vision_model.float()
        
        self.vision_model.visual.old_forward = self.vision_model.visual.forward
        if "ViT" in self.vision_encoder:
            self.vision_model.visual.forward = self.get_image_token_features
            if 'large' in self.T5_version:
                self.map_to_large = True
                self.projection = nn.Linear(512, 1024)
        else:
            self.projection = nn.Linear(2560, 512) # Use RNx4
            self.vision_model.visual.forward = self.get_resnet_features


        self.tokenizer = T5Tokenizer.from_pretrained(self.T5_version)
        self.tokenizer.add_tokens(["[itk]"])
        self.T5_model = T5ForConditionalGeneration.from_pretrained(self.T5_version)
        self.T5_model.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("[itk]")

        T5_trainable_params = 0
        vision_model_trainable_params = 0
        for para in self.T5_model.parameters():
            if para.requires_grad:
                T5_trainable_params += np.prod(para.size())
        for para in self.vision_model.parameters():
            if para.requires_grad:
                vision_model_trainable_params += np.prod(para.size())

        print(f"Initializing T5 model with {T5_trainable_params} trainable parameters ...")
        print(f"Initializing {self.vision_encoder} model with {vision_model_trainable_params} trainable parameters ...")

 

    def get_clip_text_features(self, text):
        x = self.vision_model.token_embedding(text).type(self.vision_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.vision_model.positional_embedding.type(self.vision_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vision_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.vision_model.ln_final(x).type(self.vision_model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ self.vision_model.text_projection

    def get_resnet_features(self, x):
        def stem(x):
            x = self.vision_model.visual.relu1(self.vision_model.visual.bn1(self.vision_model.visual.conv1(x)))
            x = self.vision_model.visual.relu2(self.vision_model.visual.bn2(self.vision_model.visual.conv2(x)))
            x = self.vision_model.visual.relu3(self.vision_model.visual.bn3(self.vision_model.visual.conv3(x)))
            x = self.vision_model.visual.avgpool(x)
            return x

        x = x.type(self.vision_model.visual.conv1.weight.dtype)
        x = stem(x)
        x = self.vision_model.visual.layer1(x)
        x = self.vision_model.visual.layer2(x)
        x = self.vision_model.visual.layer3(x)
        x = self.vision_model.visual.layer4(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, 3072, 49]
        x = x.permute(0, 2, 1)  # shape = [*, 49, 3072]
        x = self.projection(x)
        # x = self.attnpool(x)

        return x

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

        #x = x / x.norm(dim=1, keepdim=True)
        if self.map_to_large:
            x = self.projection(x)

        if self.use_mapping:
            x = self.mapping.linear_relu_stack(x)

        return x

    def prepare_input(self, batch):

        if self.retrieval_function:
            if self.use_quantifier:
                retrieved_info = self.retrieval_function(batch)
            else:
                retrieved_info = self.retrieval_function(batch, use_quantifier=False)
        else:
            retrieved_info = ["" for _ in batch["task"]]

        #retrieved_info = ["" for _ in batch["task"]]

        task_prefixes = [f"Answer the {x} question: " for x in batch['task']]
        # image_prompts = [" Based on the picture: " for x in batch['task']]

        image_embeddings = self.vision_model.visual(batch["image"].to(self.device))
        #image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        input_sentences = [task_prefixes[i] + batch['question'][i] + retrieved_info[i] for i in range(len(batch['question']))]
        #input_sentences = [retrieved_info[i] + task_prefixes[i] + batch['question'][i] for i in range(len(batch['question']))]
        
        encoding = self.tokenizer(
        input_sentences,
        padding="longest",
        max_length=self.max_source_length,
        truncation=True,
        return_tensors="pt",
        )

        question_embedding = self.T5_model.shared(encoding["input_ids"].to(self.device))
        #question_embedding = question_embedding / question_embedding.norm(dim=1, keepdim=True)
        
        image_attn_mask = torch.ones((image_embeddings.shape[0], image_embeddings.shape[1]))
        attention_mask = torch.cat((image_attn_mask, encoding.attention_mask), axis=1).to(self.device)
        
        if self.use_image_info: # Use image information
            combined_embedding = torch.cat((image_embeddings, question_embedding), axis=1).to(self.device)
            #combined_embedding = torch.cat((question_embedding, image_embeddings), axis=1).to(self.device)

        else: # Only use question
            combined_embedding = question_embedding.to(self.device)
            attention_mask = encoding.attention_mask.to(self.device)

        
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

    def predict(self, batch, output_attentions = False):

        combined_embedding, attention_mask, encoding = self.prepare_input(batch)

        output_sequences = self.T5_model.generate(
        inputs_embeds = combined_embedding,
        attention_mask=attention_mask,
        do_sample=False,  # disable sampling to test if batching affects output
        max_new_tokens=20
        )

        predicted_answers = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        if output_attentions:
            text = self.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
            #print(text)
            start_of_answer_span = text.index("▁certainly")
            #print(start_of_answer_span)
            output = self.T5_model(inputs_embeds = combined_embedding, labels=output_sequences, use_cache=False, output_attentions=True, return_dict=True)
            return output, start_of_answer_span
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
        

