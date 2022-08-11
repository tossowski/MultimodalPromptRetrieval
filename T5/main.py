from transformers import T5Tokenizer, T5ForConditionalGeneration
from T5VisionModel import T5VisionModel
from SLAKE import VQASLAKEFeatureDataset
from VQA_RAD import VQARADFeatureDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import torch
import argparse
import warnings
import json
import clip

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

CFG = json.load(open("config/config.json"))

data_name = CFG["dataset"]

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train a model", action="store_true")
parser.add_argument("--test", help="test a model", action="store_true")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5VisionModel().to(device)

max_source_length = CFG["max_source_length"]
max_target_length = CFG["max_target_length"]

if data_name == "VQA_RAD":
    dataset_train = VQARADFeatureDataset("train", f"data/{data_name}")
    dataset_test = VQARADFeatureDataset("test", f"data/{data_name}")
else:
    dataset_train = VQASLAKEFeatureDataset("train", f"data/{data_name}")
    dataset_test = VQASLAKEFeatureDataset("test", f"data/{data_name}")


train_loader = DataLoader(dataset_train, CFG["hyperparameters"]["batch_size"], shuffle=True, num_workers=2)
test_loader = DataLoader(dataset_test, CFG["hyperparameters"]["batch_size"], shuffle=True, num_workers=2)


learning_rate = CFG["hyperparameters"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if args.train:
    model.train()
    for batch in tqdm(train_loader):
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, f"models/model_{data_name}.pt")

# Test
if args.test:


    checkpoint = torch.load(f"models/model_{data_name}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    correct = defaultdict(int)
    performance = defaultdict(int)
    total = defaultdict(int)
    open_correct = 0
    closed_correct = 0
    open_total = 0
    closed_total = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    for batch in tqdm(test_loader):

        predicted_answers = model.predict_sequence(batch)

        for i in range(len(predicted_answers)):
            if predicted_answers[i].lower() == batch["answer"][i].lower():
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

