from T5VisionModelPredictionHeadBAN import T5VisionModelPredictionHeadBAN
from T5VisionModelPredictionHead import T5VisionModelPredictionHead
from T5VisionModel import T5VisionModel
from T5VisionModelFrozen import T5VisionModelFrozen
from SLAKE import VQASLAKEFeatureDataset
from VQA_RAD import VQARADFeatureDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from utils import get_validation_loss, visualize_attn_weights, create_ans2label

import torch
import argparse
import warnings
import json
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train a model", action="store_true")
parser.add_argument("--test", help="test a model", action="store_true")
parser.add_argument("--eval", help="test a model", action="store_true")
parser.add_argument("--config", help="Config file")
parser.add_argument("--model_file", help="path to model to save/load")
parser.add_argument("--qid", help="Question ID to analyze")
args = parser.parse_args()

CFG = json.load(open(os.path.join("config", args.config)))

data_name = CFG["dataset"]
use_image_info = bool(CFG["use_image_info"])

MODEL_SAVE_FOLDER = "./models"

if args.model_file:
    MODEL_SAVE_PATH = args.model_file
    print(f"Model will be saved/loaded from {MODEL_SAVE_PATH}")
else:
    MODEL_PREFIX = f"model_{data_name}"
    if use_image_info:
        MODEL_PREFIX += "_with_vision"
    else:
        MODEL_PREFIX += "_no_vision"

    if CFG["vision_checkpoint"]:
        MODEL_PREFIX += "_with_pretrained_checkpoint"
    else:
        MODEL_PREFIX += "_no_pretrained_checkpoint"

    if CFG["fewshot_training_tasks"]["enabled"]:
        MODEL_PREFIX += "_fewshot"

    if CFG["mapping_checkpoint"]:
        MODEL_PREFIX += "_with_mapping"
    else:
        MODEL_PREFIX += "_no_mapping"

    if CFG["use_prediction_head"]:
        if CFG["use_BAN"]:
            MODEL_PREFIX += "_pred_headBAN"
        else:
            MODEL_PREFIX += "_pred_head"
    else:
        MODEL_PREFIX += ""

    if CFG["freeze"]:
        MODEL_PREFIX += "_freeze"

    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_FOLDER, MODEL_PREFIX + ".pt")
    print(f"Model will be saved/loaded from {MODEL_SAVE_PATH}")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
max_source_length = CFG["max_source_length"]
max_target_length = CFG["max_target_length"]

torch.manual_seed(CFG["seed"])

if data_name == "VQA_RAD":
    dataset_train = VQARADFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device) 
    # VQA_RAD doesn't have validation data, so use subset of train to estimate
    
    # train_split = list(range(len(dataset_train) // 8, len(dataset_train)))
    # validate_split = list(range(0, len(dataset_train) // 8))

    # dataset_train = torch.utils.data.Subset(dataset_train, train_split)
    # dataset_validate = torch.utils.data.Subset(dataset_train, validate_split)
    
    dataset_validate = VQARADFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device) 
    #dataset_validate.entries = dataset_validate.entries[0:len(dataset_train) // 8]
    #dataset_train.entries = dataset_train.entries[len(dataset_train) // 8:]
    #dataset_validate = VQARADFeatureDataset("test", os.path.join(CFG["datafolder"],data_name), device=device) 
    
    dataset_test = VQARADFeatureDataset("test", os.path.join(CFG["datafolder"],data_name), device=device)
else:
    dataset_train = VQASLAKEFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device)
    dataset_validate = VQASLAKEFeatureDataset("validate", os.path.join(CFG["datafolder"],data_name), device=device)
    dataset_test = VQASLAKEFeatureDataset("test", os.path.join(CFG["datafolder"],data_name), device=device)

if CFG["max_answers"]:
    answer_set = dataset_train.filter_max_answers(CFG["max_answers"], config=CFG)
    dataset_validate.filter_max_answers(CFG["max_answers"], answer_set)
    dataset_test.filter_max_answers(CFG["max_answers"], answer_set)
    print(f"Train data has {len(dataset_train)} examples\nValidation data has {len(dataset_validate)} examples\nTest data has {len(dataset_test)} examples")
    


label2ans, ans2label = create_ans2label(dataset_train, dataset_validate, dataset_test)    
dataset_train.add_labels(ans2label)
dataset_validate.add_labels(ans2label)
dataset_test.add_labels(ans2label)
print(ans2label)

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

if CFG["use_prediction_head"]:
    if CFG["use_BAN"]:
        model = T5VisionModelPredictionHeadBAN(len(ans2label), vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"], glimpse=CFG["glimpse"]).to(device)
    else:
        if CFG["max_answers"]:
            model = T5VisionModelPredictionHead(CFG["max_answers"], vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"]).to(device)
        else:
            model = T5VisionModelPredictionHead(len(ans2label), vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"]).to(device)

else:
    if CFG["freeze"]:
        model = T5VisionModelFrozen(vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"]).to(device)   
    else:
        model = T5VisionModel(vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"]).to(device)


learning_rate = CFG["hyperparameters"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

if args.train:
    best_valid_loss = float("inf")
    best_epoch = 0
    for epoch in range(CFG["hyperparameters"]["epochs"]):
        #model.train()
        print(f"Starting epoch {epoch} ...")
        train_total = 0
        train_batch = 0
        total_ans = 0
        total_correct_ans = 0
        for batch in tqdm(train_loader):
            loss = model(batch)
            pred = model.predict(batch)

            if CFG["use_prediction_head"]:
                total_correct_ans += torch.sum(torch.eq(batch["labels"].to(model.device), pred))
                total_ans += len(batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item()
        if CFG["use_prediction_head"]:
            print(f"Train acc is: {total_correct_ans / total_ans}")
        else:
            print(f"Train loss is {train_total / len(train_loader.dataset)}")
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
        predicted_answers = model.predict(batch)


        
        
        for i in range(len(predicted_answers)):
            if CFG["use_prediction_head"]:
                is_correct = predicted_answers[i] == batch["labels"][i]
            else:
                is_correct = predicted_answers[i].lower() == batch["answer"][i].lower()
            
            if is_correct:
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
                #print(batch["question"][i], label2ans[predicted_answers[i].item()])
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

    with open(os.path.join("logs", "200EPOCH_NO_MODEL_TRAIN_" + MODEL_PREFIX + ".txt"), "w") as f:
        for key, val in performance.items():
            f.write(f"{key},{round(val, 2)}\n")
        f.write(f"Open,{round(open_correct/open_total, 2)}\n")
        f.write(f"Closed: {round(closed_correct/closed_total, 2)}\n")
        f.write(f"Overall,{round(sum(correct.values())/sum(total.values()), 2)}")

if args.eval:
    checkpoint = torch.load(MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()
    with open(os.path.join("logs", "correct_ids.txt"), "r") as f:
        num_lines = sum(1 for line in f if line.rstrip())
        
    with open(os.path.join("logs", "correct_ids.txt"), "r") as f:
        for i, line in enumerate(f):
            qid = line[:-1]

            #if qid == "12098":
            info = dataset_test.get_question_by_id(qid)
            batch = test_loader.collate_fn([info])

            visualize_attn_weights(model, batch, attn_type = "cross_attentions")
            print(f"Finished image {i} out of {num_lines}")