import clip
import torch
import os
import numpy as np
from PIL import Image
from scipy.spatial import distance
from SLAKE import VQASLAKEFeatureDataset
from VQA_RAD import VQARADFeatureDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from T5VisionModel import T5VisionModel

import sys

import json
import argparse
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Config file")
args = parser.parse_args()

CFG = json.load(open(os.path.join("config", args.config)))

device = "cuda:1" if torch.cuda.is_available() else "cpu"
device = "cpu"
data_name = CFG["dataset"]

if data_name == "VQA_RAD":
    dataset_train = VQARADFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device)
    dataset_test = VQARADFeatureDataset("test", os.path.join(CFG["datafolder"],data_name), device=device)
else:
    dataset_train = VQASLAKEFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device)
    dataset_test = VQASLAKEFeatureDataset("test", os.path.join(CFG["datafolder"],data_name), device=device)
train_loader = DataLoader(dataset_train, 1, shuffle=False, num_workers=1)
test_loader = DataLoader(dataset_test, 1, shuffle=False, num_workers=1)


model, preprocess = clip.load("ViT-B/32", device=device)
model = model.float()
image_features = []
answers = {}
text_features = []
text_features_test = []
test_image_features = {}
image_to_questions = {}
image_features = {}

idx = 0
image_features_path_train = f"./image_features_{data_name}_train.pkl"
text_features_path_train = f"./text_features_{data_name}_train.pkl"
image_features_path_test = f"./image_features_{data_name}_test.pkl"
text_features_path_test = f"./text_features_{data_name}_test.pkl"
if os.path.exists(image_features_path_train) and os.path.exists(text_features_path_train) and os.path.exists(image_features_path_test):
    print("Loading existing train image features ...")
    with open(image_features_path_train, 'rb') as f:
        image_features = pickle.load(f)
    print(f"Loaded {len(image_features.keys())} existing train image features")
    
    print("Loading existing train text features ...")
    with open(text_features_path_train, 'rb') as f:
        text_features = pickle.load(f)
    print(f"Loaded {len(text_features)} existing train text features")

    print("Loading existing test image features ...")
    with open(image_features_path_test, 'rb') as f:
        test_image_features = pickle.load(f)
    print(f"Loaded {len(test_image_features.keys())} existing test image features")
    
    print("Loading existing test text features ...")
    with open(text_features_path_test, 'rb') as f:
        text_features_test = pickle.load(f)
    print(f"Loaded {len(text_features_test)} existing test text features")
    # Create image to questions mapping
    for batch in tqdm(train_loader):
        image_path = batch["path_to_image"][0]
        if image_path not in image_to_questions:
            image_to_questions[image_path] = []
        image_to_questions[image_path].append(idx)
        answers[idx] = batch["answer"][0]
        idx += 1
else:
    print("Computing Features ...")
    # for batch in tqdm(train_loader):
    #     image_path = batch["path_to_image"][0]
    #     if image_path not in image_features:
    #         image_features[image_path] = model.encode_image(batch["image"].to(device))
        
    #     if image_path not in image_to_questions:
    #         image_to_questions[image_path] = []
    #     image_to_questions[image_path].append(idx)


    #     tokenized = clip.tokenize(batch["question"]).to(device)
    #     text_features.append(model.encode_text(tokenized))
    #     answers[idx] = batch["answer"][0]

    #     idx += 1


    for batch in tqdm(test_loader):
        image_path = batch["path_to_image"][0]
        if image_path not in test_image_features:
            test_image_features[image_path] = model.encode_image(batch["image"].to(device))
        tokenized = clip.tokenize(batch["question"]).to(device)
        text_features_test.append(model.encode_text(tokenized))

    # with open(image_features_path_train, 'wb') as f:
    #     pickle.dump(image_features, f)
    
    # with open(text_features_path_train, 'wb') as f:
    #     pickle.dump(text_features, f)
    
    with open(image_features_path_test, 'wb') as f:
        pickle.dump(test_image_features, f)

    with open(text_features_path_test, 'wb') as f:
        pickle.dump(text_features_test, f)


model_path = "models/model_SLAKE_with_vision_with_pretrained_checkpoint_no_mapping.pt"
T5_model = T5VisionModel(vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=True, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"]).to(device)
optimizer = torch.optim.AdamW(T5_model.parameters(), lr=0.00001)
checkpoint = torch.load(model_path)
T5_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

all_questions = []
for batch in tqdm(train_loader):
    all_questions.append(batch["question"][0])

confidence = 0

fig = plt.figure(figsize=(15,10))
image_feat_pairs = sorted([(k, v) for k, v in image_features.items()], key = lambda x: x[0]) # Sort image, feat pairs alphabetically
images = []
for pair in image_feat_pairs:
    images.append(pair[1])
images = torch.cat(images, axis=0)
images = images.detach().cpu().numpy()
all_dists = []

for confidence in np.linspace(0, 0.9, 10):
    accs = []
    for n_neighbors in range(1, 31):
        correct = 0
        total = 0
        super_correct = 0
        pred_in_answers = 0

        neighbors = n_neighbors
        for t_idx, batch in enumerate(test_loader):
            i_feature = test_image_features[batch["path_to_image"][0]]
            dist_images = distance.cdist(i_feature.detach().cpu().numpy(), images)
            sorted_dist_images = np.argsort(dist_images, axis = 1)
            nearest_images_idx = sorted_dist_images[0, 1 : neighbors + 1]
            closest_image_pairs = [image_feat_pairs[x] for x in nearest_images_idx]
            closest_image_names = [x[0] for x in closest_image_pairs]
            current_image = batch["path_to_image"][0]
            

            # GO through each question in current image and answer it
            current_question_feature = text_features_test[t_idx].detach().cpu().numpy()
            related_questions = []

            for i_name in closest_image_names:
                rel_image_questions = image_to_questions[i_name]
                question_features = [text_features[i] for i in rel_image_questions]
                question_features = torch.cat(question_features, axis = 0)
                question_features = question_features.detach().cpu().numpy()
                question_dists = distance.cdist(current_question_feature, question_features, metric='cosine')
                sorted_question_dists = np.argsort(question_dists, axis=1)
                all_dists.append(question_dists[0][sorted_question_dists[0][0]])
                # if question_dists[0][sorted_question_dists[0][0]] > 0.05:
                #     continue
                #     print(all_questions[rel_image_questions[sorted_question_dists[0][0]]], batch["question"])
                related_questions.append(rel_image_questions[sorted_question_dists[0][0]])
            
            ordered_answers = [answers[x] for x in related_questions]
            answer_counts = {}
            for answer in ordered_answers:
                if answer not in answer_counts:
                    answer_counts[answer] = 0
                answer_counts[answer] += 1
            if answer_counts == {}:
                continue
            pred_answer = max(answer_counts, key = answer_counts.get)
            if answer_counts[pred_answer] / sum(answer_counts.values()) < confidence:
                continue
            
            T5_model_prediction = T5_model.predict(batch)[0]
            if batch["answer"][0] in answer_counts:
                pred_in_answers += 1
            if pred_answer == batch["answer"][0] or T5_model_prediction == batch["answer"][0]:
                if T5_model_prediction != pred_answer and batch["answer"][0] == pred_answer:
                    super_correct += 1
                correct += 1
            total += 1
        print(f"The acc for confidence level {round(confidence, 1)} and {n_neighbors} nearest neighbors is {correct/total} out of {total}")
        print(f"There were {super_correct} times where nn predicted correct but T5 didn't out of {total}")
        print(f"THe upper bound is {pred_in_answers/total}")
        accs.append(correct/total)

    plt.plot(list(range(1,31)), accs, label=f"Confidence {round(confidence, 1)}")
plt.title("Nearest Neighbor Accuracy")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accs.png")

# plt.hist(all_dists)
# plt.savefig("dists")
