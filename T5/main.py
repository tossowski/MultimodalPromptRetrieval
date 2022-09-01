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
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

CFG = json.load(open("config/config.json"))


data_name = CFG["dataset"]
use_image_info = bool(CFG["use_image_info"])

MODEL_SAVE_FOLDER = "./models"
MODEL_PREFIX = f"model_{data_name}"
if use_image_info:
    MODEL_PREFIX += "_with_vision"
else:
    MODEL_PREFIX += "_no_vision"

if CFG["fewshot_training_tasks"]["enabled"]:
    MODEL_PREFIX += "_fewshot"

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_FOLDER, MODEL_PREFIX + ".pt")
print(f"Model will be saved/loaded from {MODEL_SAVE_PATH}")

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train a model", action="store_true")
parser.add_argument("--test", help="test a model", action="store_true")
parser.add_argument("--eval", help="test a model", action="store_true")
parser.add_argument("--qid", help="Question ID to analyze")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5VisionModel(vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"]).to(device)

max_source_length = CFG["max_source_length"]
max_target_length = CFG["max_target_length"]

torch.manual_seed(CFG["seed"])

if data_name == "VQA_RAD":
    dataset_train = VQARADFeatureDataset("train", os.path.join(CFG["datafolder"],data_name)) 
    # VQA_RAD doesn't have validation data, so use subset of train to estimate
    
    train_split = list(range(len(dataset_train) // 8, len(dataset_train)))
    validate_split = list(range(0, len(dataset_train) // 8))

    dataset_train = torch.utils.data.Subset(dataset_train, train_split)
    dataset_validate = torch.utils.data.Subset(dataset_train, validate_split)
    dataset_test = VQARADFeatureDataset("test", os.path.join(CFG["datafolder"],data_name))
else:
    dataset_train = VQASLAKEFeatureDataset("train", os.path.join(CFG["datafolder"],data_name))
    dataset_validate = VQASLAKEFeatureDataset("validate", os.path.join(CFG["datafolder"],data_name))
    dataset_test = VQASLAKEFeatureDataset("test", os.path.join(CFG["datafolder"],data_name))

if CFG["fewshot_training_tasks"]["enabled"]:
    #tasks = list(set([x["task"] for x in dataset_train.entries]))
    test_tasks = CFG["fewshot_training_tasks"]["test"]
    training_tasks = CFG["fewshot_training_tasks"]["train"]
    
    print(f'Filtering training to only consist of these tasks: {training_tasks}')
    dataset_train.filter(training_tasks, limit_num_examples = CFG["fewshot_training_tasks"]["limit"])
    dataset_validate.filter(training_tasks, limit_num_examples = CFG["fewshot_training_tasks"]["limit"])
    print(f'Filtering test to only consist of these tasks: {test_tasks}')
    dataset_test.filter(test_tasks, limit_num_examples = CFG["fewshot_training_tasks"]["limit"])

print(f"Train data has {len(dataset_train)} examples\nValidation data has {len(dataset_validate)} examples\nTest data has {len(dataset_test)} examples")

train_loader = DataLoader(dataset_train, CFG["hyperparameters"]["batch_size"], shuffle=True, num_workers=2)
validate_loader = DataLoader(dataset_validate, CFG["hyperparameters"]["batch_size"], shuffle=True, num_workers=2)
test_loader = DataLoader(dataset_test, CFG["hyperparameters"]["batch_size"], shuffle=True, num_workers=2)


learning_rate = CFG["hyperparameters"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

if args.train:
    best_valid_loss = float("inf")
    best_epoch = 0
    for epoch in range(CFG["hyperparameters"]["epochs"]):
        #model.train()
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
            os.makedirs(MODEL_SAVE_FOLDER, exist_ok = True)
            torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, MODEL_SAVE_PATH)
            best_valid_loss = valid_loss
            best_epoch = epoch

# Test
if args.test:
    #MODEL_SAVE_PATH = "./models/model_SLAKE_with_vision.pt" # For OOD Testing
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()

    correct = defaultdict(int)
    performance = defaultdict(int)
    total = defaultdict(int)
    open_correct = 0
    closed_correct = 0
    open_total = 0
    closed_total = 0

    incorrect_ids = []
    correct_ids = []
    for batch in tqdm(test_loader):
        predicted_answers = model.predict_sequence(batch)

        
        
        for i in range(len(predicted_answers)):
            if predicted_answers[i].lower() == batch["answer"][i].lower():
                #print(f'{batch["question"][i]} ||| {predicted_answers[i]} ||| {batch["answer"][i]}')
                correct_ids.append(batch["question_id"][i])
                correct[batch["task"][i]] += 1
                if batch["question_type"][i] == "open":
                    open_correct += 1
                else:
                    closed_correct += 1
            else:
                incorrect_ids.append(batch["question_id"][i])

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
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "incorrect_ids.txt"), "w") as f:
        for qid in incorrect_ids:
            f.write(qid + "\n")
    
    with open(os.path.join("logs", "correct_ids.txt"), "w") as f:
        for qid in correct_ids:
            f.write(qid + "\n")


if args.eval:
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()
    with open(os.path.join("logs", "incorrect_ids.txt"), "r") as f:
        num_lines = sum(1 for line in f if line.rstrip())
        
    with open(os.path.join("logs", "incorrect_ids.txt"), "r") as f:
        for i, line in enumerate(f):
            qid = line[:-1]

            info = dataset_test.get_question_by_id(qid)
            batch = test_loader.collate_fn([info])

            visualize_attn_weights(model, batch, attn_type = "cross_attentions")
            print(f"Finished image {i} out of {num_lines}")
    #break