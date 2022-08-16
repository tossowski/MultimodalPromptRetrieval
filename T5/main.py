from T5VisionModel import T5VisionModel
from SLAKE import VQASLAKEFeatureDataset
from VQA_RAD import VQARADFeatureDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from utils import get_validation_loss, visualize_attn_weights

import torch
import argparse
import warnings
import json

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

CFG = json.load(open("config/config.json"))

data_name = CFG["dataset"]
use_image_info = bool(CFG["use_image_info"])

if use_image_info:
    MODEL_SAVE_PATH = f"models/model_{data_name}_with_vision.pt"
else:
    MODEL_SAVE_PATH = f"models/model_{data_name}_no_vision.pt"


parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train a model", action="store_true")
parser.add_argument("--test", help="test a model", action="store_true")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5VisionModel(use_image_info=use_image_info).to(device)

max_source_length = CFG["max_source_length"]
max_target_length = CFG["max_target_length"]

if data_name == "VQA_RAD":
    dataset_train = VQARADFeatureDataset("train", f"data/{data_name}") 
    # VQA_RAD doesn't have validation data, so use subset of train to estimate
    
    train_split = list(range(len(dataset_train) // 8, len(dataset_train)))
    validate_split = list(range(0, len(dataset_train) // 8))

    dataset_train = torch.utils.data.Subset(dataset_train, train_split)
    dataset_validate = torch.utils.data.Subset(dataset_train, validate_split)
    dataset_test = VQARADFeatureDataset("test", f"data/{data_name}")
else:
    dataset_train = VQASLAKEFeatureDataset("train", f"data/{data_name}")
    dataset_validate = VQASLAKEFeatureDataset("validate", f"data/{data_name}")
    dataset_test = VQASLAKEFeatureDataset("test", f"data/{data_name}")


train_loader = DataLoader(dataset_train, CFG["hyperparameters"]["batch_size"], shuffle=True, num_workers=2)
validate_loader = DataLoader(dataset_validate, CFG["hyperparameters"]["batch_size"], shuffle=True, num_workers=2)
test_loader = DataLoader(dataset_test, CFG["hyperparameters"]["batch_size"], shuffle=True, num_workers=2)


learning_rate = CFG["hyperparameters"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

if args.train:
    model.train()
    best_valid_loss = float("inf")
    best_epoch = 0
    for epoch in range(CFG["hyperparameters"]["epochs"]):
        print(f"Starting epoch {epoch} ...")
        for batch in tqdm(train_loader):
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_loss = get_validation_loss(model, validate_loader)
        scheduler.step(valid_loss)

        print(f"Validation Loss: {valid_loss} | Best Validation Loss: {best_valid_loss} at epoch {best_epoch}")
        if valid_loss < best_valid_loss:
            print(f"Saving model to {MODEL_SAVE_PATH} ...")
            torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, MODEL_SAVE_PATH)
            best_valid_loss = valid_loss
            best_epoch = epoch

# Test
if args.test:


    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    correct = defaultdict(int)
    performance = defaultdict(int)
    total = defaultdict(int)
    open_correct = 0
    closed_correct = 0
    open_total = 0
    closed_total = 0

    for batch in tqdm(test_loader):

        predicted_answers = model.predict_sequence(batch)
        #visualize_attn_weights(model, batch)
        #break

        for i in range(len(predicted_answers)):
            if predicted_answers[i].lower() == batch["answer"][i].lower():
                print(f'{batch["question"][i]} ||| {predicted_answers[i]} ||| {batch["answer"][i]}')
                
                correct[batch["task"][i]] += 1
                if batch["question_type"][i] == "open":
                    open_correct += 1
                else:
                    closed_correct += 1

            total[batch["task"][i]] += 1
            if batch["question_type"][i] == "open":
                open_total += 1
            else:
                closed_total += 1

    for key in correct:
        performance[key] = correct[key] / total[key]


    print("=======QUESTION TYPE PERFORMANCE=======")
    for key, val in performance.items():
        print(f"{key}: {round(val, 2)}")
    print("=======OPEN VS CLOSED PERFORMANCE======")
    print(f"Open: {round(open_correct/open_total, 2)}")
    print(f"Closed: {round(closed_correct/closed_total, 2)}")
    print("===========OVERALL PERFORMANCE=========")
    print(f"Overall accuracy: {round(sum(correct.values())/sum(total.values()), 2)}")
        #print([batch['answer'][i] + "|||" + predicted_answers[i] for i in range(len(predicted_answers))])

