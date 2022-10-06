

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from T5VisionModel import T5VisionModel
from create_mapping import CrossModalMapping
from network.connect import FCNet
from network.connect import BCNet
from torch.nn.utils.weight_norm import weight_norm

# Bilinear Attention
class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):  #128, 1024, 1024,2
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):  # v:32,1,128; q:32,12,1024
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q
        #print(v_num, q_num, logits.shape)

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            #print(mask.shape, v.shape)
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits


class BiResNet(nn.Module):
    def __init__(self, v_dim, q_dim, glimpse):
        super(BiResNet,self).__init__()
        # Optional module: counter
        # use_counter = cfg.TRAIN.ATTENTION.USE_COUNTER if priotize_using_counter is None else priotize_using_counter
        # if use_counter or priotize_using_counter:
        #     objects = 10  # minimum number of boxes
        # if use_counter or priotize_using_counter:
        #     counter = Counter(objects)
        # else:
        #     counter = None
        # # init Bilinear residual network
        self.glimpse = glimpse
        b_net = []   # bilinear connect :  (XTU)T A (YTV)
        q_prj = []   # output of bilinear connect + original question-> new question    Wq_ +q
        c_prj = []
        for i in range(self.glimpse):
            b_net.append(BCNet(v_dim, q_dim, q_dim, None, k=1))
            q_prj.append(FCNet([q_dim, q_dim], '', .2))
            # if use_counter or priotize_using_counter:
            #     c_prj.append(FCNet([objects + 1, cfg.TRAIN.QUESTION.HID_DIM], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)


    def forward(self, v_emb, q_emb, att_p):
        b_emb = [0] * self.glimpse
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att_p[:,g,:,:]) # b x l x h
            # atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        #print(q_emb.shape, q_emb.sum(1).shape)
        return q_emb.sum(1)


class T5VisionModelPredictionHeadBAN(T5VisionModel):
    def __init__(self, num_classes, vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128, use_image_info=True, vision_checkpoint=None, mapping_checkpoint=None, glimpse = 10):
        super().__init__(vision_encoder = "ViT-B/32", T5_version = "t5-small", max_source_length = 512, max_target_length = 128, use_image_info=True, vision_checkpoint=None, mapping_checkpoint=None)
        self.loss_fn = torch.nn.CrossEntropyLoss()
 
        self.num_classes = num_classes
        self.BAN_att = BiAttention(512, 512, 512, 10)
        self.BAN_resnet = BiResNet(512, 512, 10)
        self.prediction_head = nn.Linear(512, self.num_classes)
        self.prediction_dropout = nn.Dropout(0.1) 

    def predict(self, batch):

        question_embedding, image_embeddings, attention_mask, _ = self.prepare_input(batch)
        
        target_encoding = self.tokenizer(
        batch['answer'], padding="longest", max_length=self.max_target_length, truncation=True
        )

        labels = target_encoding.input_ids
        labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels = labels.to(self.device)

        
        output = self.T5_model(inputs_embeds = question_embedding, attention_mask=attention_mask, labels=labels)
        #print(output['encoder_last_hidden_state'].shape)
        question_features = output['encoder_last_hidden_state']
        image_features = image_embeddings 
        attn, _ = self.BAN_att(image_features, question_features)
        resnet_output = self.BAN_resnet(image_features,question_features,attn)
        dropped = self.prediction_dropout(resnet_output)
        logits = self.prediction_head(dropped)
        predictions = torch.argmax(logits, dim = 1)
        #probs = logits.softmax(dim = 1)
        #print(batch['labels'])
        return predictions


    def prepare_input(self, batch):
        task_prefixes = [f"Answer the {x} question: " for x in batch['task']]
        image_embeddings = self.vision_model.visual(batch["image"].to(self.device))
    
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
        question_embedding.to(self.device)
        
        norm = question_embedding.pow(2).sum(keepdim=True, dim=2).sqrt()
        question_embedding = question_embedding / norm
        norm = image_embeddings.pow(2).sum(keepdim=True, dim=2).sqrt()
        image_embeddings = image_embeddings / norm
        # PCA CODE

        # data = combined_embedding.detach().cpu().numpy()[0]

        # scaler = StandardScaler()
        # scaler.fit(data)
        # data=scaler.transform(data)    
        # labels = np.zeros(data.shape[0])
        # labels[-51:-1] = 1

        # pca = PCA()
        # x_new = pca.fit_transform(data)
        # my_colors = np.where(labels == 1, "red", "blue")
        # fig = plt.figure()
        # plt.scatter(x_new[:,0], x_new[:,1],color=my_colors)
        # plt.savefig(f"pca_{batch['question_id'][0]}.png")

        # test = combined_embedding.detach().cpu().numpy()
        # fig, ax = plt.subplots(1, len(test[0]), figsize=(200,10))
        # for i in range(len(test[0])):
        #     ax[i].hist(test[0][i])
        # plt.savefig("test.png")
        
        return question_embedding, image_embeddings, attention_mask, encoding
    def forward(self, batch):

        question_embedding, image_embeddings, attention_mask, _ = self.prepare_input(batch)
        
        target_encoding = self.tokenizer(
        batch['answer'], padding="longest", max_length=self.max_target_length, truncation=True
        )

        labels = target_encoding.input_ids
        labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels = labels.to(self.device)

        
        #print(batch['labels'])
        output = self.T5_model(inputs_embeds = question_embedding, attention_mask=attention_mask, labels=labels)
        #print(output['encoder_last_hidden_state'].shape)
        question_features = output['encoder_last_hidden_state']
        image_features = image_embeddings 
        attn, _ = self.BAN_att(image_features, question_features)
        resnet_output = self.BAN_resnet(image_features,question_features,attn)
        dropped = self.prediction_dropout(resnet_output)
        logits = self.prediction_head(dropped)
        return self.loss_fn(logits, batch['labels'].to(self.device))
        

