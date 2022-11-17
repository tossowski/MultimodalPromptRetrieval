from math import sqrt
from transformers import T5Tokenizer, T5ForConditionalGeneration
import clip
import torch
from torch import nn
from create_mapping import CrossModalMapping
from utils import cosine_similarity
from torch.autograd import Variable

class T5VisionModel(nn.Module):
    def __init__(self, vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128, use_image_info=True, vision_checkpoint=None, mapping_checkpoint=None, retrieval_function=None):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.device = "cpu"
        self.vision_encoder = vision_encoder
        self.T5_version = T5_version
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.use_image_info = use_image_info
        self.retrieval_function = retrieval_function
        self.use_mapping = bool(mapping_checkpoint)
        self.vision_model, _ = clip.load(self.vision_encoder, device=self.device)
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
        self.vision_model.visual.forward = self.get_image_token_features

        self.tokenizer = T5Tokenizer.from_pretrained(self.T5_version)
        self.tokenizer.add_tokens(["[itk]"])
        self.T5_model = T5ForConditionalGeneration.from_pretrained(self.T5_version)
        self.T5_model.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("[itk]")

    def attention_fn(self, query, context, temp1):
        """
        query: batch x ndf x queryL
        context: batch x ndf x ih x iw (sourceL=ihxiw)
        mask: batch_size x sourceL
        """
        batch_size, queryL = query.size(0), query.size(2)
        ih, iw = context.size(2), context.size(3)
        sourceL = ih * iw

        # --> batch x sourceL x ndf
        context = context.view(batch_size, -1, sourceL)
        contextT = torch.transpose(context, 1, 2).contiguous()

        # Get attention
        # (batch x sourceL x ndf)(batch x ndf x queryL)
        # -->batch x sourceL x queryL
        attn = torch.bmm(contextT, query)
        # --> batch*sourceL x queryL
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax(dim=-1)(attn)

        # --> batch x sourceL x queryL
        attn = attn.view(batch_size, sourceL, queryL)
        # --> batch*queryL x sourceL
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = attn.view(batch_size * queryL, sourceL)

        attn = attn * temp1
        attn = nn.Softmax(dim=-1)(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # (batch x ndf x sourceL)(batch x sourceL x queryL)
        # --> batch x ndf x queryL
        weightedContext = torch.bmm(context, attnT)

        return weightedContext, attn.view(batch_size, -1, ih, iw)

    def local_loss(
        self, words_emb, img_features, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"):

        batch_size = img_features.shape[0]

        att_maps = []
        similarities = []
        # cap_lens = cap_lens.data.tolist()
        for i in range(words_emb.shape[0]):

            # Get the i-th text description
            words_num = cap_lens[i]  # 25
            # TODO: remove [SEP]
            # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
            word = words_emb[i, :words_num, :].unsqueeze(0).transpose(1,2).contiguous()  # [1, 768, 25]
            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            grid_size = int(sqrt(img_features.shape[1]))
            context = img_features.view(batch_size, img_features.shape[-1], grid_size, grid_size)  # [48, 768, 19, 19]
                        
            weiContext, attn = self.attention_fn(
                word, context, temp1
            )  # [48, 768, 25], [48, 25, 19, 19]

            att_maps.append(
                attn[i].unsqueeze(0).contiguous()
            )  # add attention for curr index  [25, 19, 19]
            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

            row_sim = cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(temp2).exp_()
            if agg == "sum":
                row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
            else:
                row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        similarities = torch.cat(similarities, 1)  #
        similarities = similarities * temp3
        similarities1 = similarities.transpose(0, 1)  # [48, 48]

        labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)

        loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
        return loss0, loss1, att_maps

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

        if self.use_mapping:
            x = self.mapping.linear_relu_stack(x)

        return x

    def prepare_input(self, batch):

        if self.retrieval_function:
            retrieved_info = self.retrieval_function(batch)
        else:
            retrieved_info = ["" for _ in batch["task"]]

        task_prefixes = [f"Answer the {x} question: " for x in batch['task']]
        # image_prompts = [" Based on the picture: " for x in batch['task']]

        image_embeddings = self.vision_model.visual(batch["image"].to(self.device))

        # image_tokens = "[itk] " * image_embeddings.shape[1]
        # image_tokens = image_tokens[:-1] # Remove last space
    
  
        input_sentences = [task_prefixes[i] + batch['question'][i] + retrieved_info[i] for i in range(len(batch['question']))]
        
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
        #print(image_attn_mask, encoding.attention_mask)
        #print(image_attn_mask.shape, encoding.attention_mask.shape)
        attention_mask = torch.cat((image_attn_mask, encoding.attention_mask), axis=1).to(self.device)
        #print(attention_mask.shape)
        if self.use_image_info:
            combined_embedding = torch.cat((image_embeddings, question_embedding), axis=1).to(self.device)
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

    def predict(self, batch):

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

        # caption_encoding = self.tokenizer(
        #     batch['question'],
        #     padding="longest",
        #     max_length=self.max_source_length,
        #     truncation=True,
        #     return_tensors="pt"
        # )

        #lengths = [sum([1 for entry in caption_encoding["input_ids"][i] if entry != 0]) for i in range(len(caption_encoding["input_ids"]))]
        #image_embeddings = self.vision_model.visual(batch["image"].to(self.device))
        #caption_embedding = self.T5_model.shared(caption_encoding["input_ids"].to(self.device))
        #loss0, loss1, attn_maps = self.local_loss(caption_embedding, image_embeddings[:,1:,:], lengths) # Get everything but CLS token
        #print(len(attn_maps))
        #print(attn_maps[0].shape)

        target_encoding = self.tokenizer(
        batch['answer'], padding="longest", max_length=self.max_target_length, truncation=True
        )

        labels = target_encoding.input_ids
        labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels = labels.to(self.device)


        loss = self.T5_model(inputs_embeds = combined_embedding, attention_mask=attention_mask, labels=labels).loss
        #loss = loss
        return loss
        

