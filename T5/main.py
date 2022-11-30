from T5VisionModelPredictionHeadBAN import T5VisionModelPredictionHeadBAN
from T5VisionModelPredictionHead import T5VisionModelPredictionHead
from T5VisionModel import T5VisionModel
from T5VisionModelFrozen import T5VisionModelFrozen
from SLAKE import VQASLAKEFeatureDataset
from VQA_RAD import VQARADFeatureDataset
from ROCO import ROCOFeatureDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from utils import get_validation_loss, visualize_attn_weights, create_ans2label, get_model_prefix

import torch
import argparse
import warnings
import json
import os
import random

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train a model", action="store_true")
parser.add_argument("--resume", help="Resume model training", action="store_true")
parser.add_argument("--test", help="test a model", action="store_true")
parser.add_argument("--eval", help="evaluate a model", action="store_true")
parser.add_argument("--config", help="config file name in the config folder")
parser.add_argument("--model_file", help="optional path to model to save/load")
parser.add_argument("--qid", help="Question ID to analyze")
args = parser.parse_args()



CFG = json.load(open(os.path.join("config", args.config)))
random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed(CFG["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

data_name = CFG["dataset"]
use_image_info = bool(CFG["use_image_info"])
MODEL_SAVE_FOLDER = "./models"

if args.model_file:
    MODEL_SAVE_PATH = args.model_file
    print(f"Model will be saved/loaded from {MODEL_SAVE_PATH}")
else:
    MODEL_PREFIX = get_model_prefix(CFG)

    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_FOLDER, MODEL_PREFIX + ".pt")
    print(f"Model will be saved/loaded from {MODEL_SAVE_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"
max_source_length = CFG["max_source_length"]
max_target_length = CFG["max_target_length"]

torch.manual_seed(CFG["seed"])

if "transfer_dataset" in CFG:
    print(f"Evaluating on transfer dataset {CFG['transfer_dataset']}")
    data_name = CFG["transfer_dataset"]

if data_name == "VQA_RAD":
    dataset_train = VQARADFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device) 
    # VQA_RAD doesn't have validation data, so use subset of train to estimate
    
    # train_split = list(range(len(dataset_train) // 8, len(dataset_train)))
    # validate_split = list(range(0, len(dataset_train) // 8))

    # dataset_train = torch.utils.data.Subset(dataset_train, train_split)
    # dataset_validate = torch.utils.data.Subset(dataset_train, validate_split)
    length = len(dataset_train)
    dataset_validate = VQARADFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device) 
    
    #random_split = random.sample(list(range(len(dataset_train))), len(dataset_train) // 10)
    #random_split = dataset_validate.get_stratified_split(0.15, CFG["seed"])

    #dataset_validate.entries = [dataset_train.entries[i] for i in range(length) if i in random_split]
    #dataset_train.entries = [dataset_train.entries[i] for i in range(length) if i not in random_split]
    #print(dataset_validate)
    #print(dataset_train)
    #dataset_validate = VQARADFeatureDataset("test", os.path.join(CFG["datafolder"],data_name), device=device) 
    
    dataset_test = VQARADFeatureDataset("test", os.path.join(CFG["datafolder"],data_name), device=device)
elif data_name == "SLAKE":
    dataset_train = VQASLAKEFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device)
    dataset_validate = VQASLAKEFeatureDataset("validate", os.path.join(CFG["datafolder"],data_name), device=device)
    dataset_test = VQASLAKEFeatureDataset("test", os.path.join(CFG["datafolder"],data_name), device=device)
elif data_name == "ROCO":
    dataset_train = ROCOFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device)
    dataset_validate = ROCOFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device)
    dataset_test = ROCOFeatureDataset("train", os.path.join(CFG["datafolder"],data_name), device=device)

if "max_answers" in CFG and CFG["max_answers"]:
    answer_set = dataset_train.filter_max_answers(CFG["max_answers"], config=CFG)
    dataset_validate.filter_max_answers(CFG["max_answers"], answer_set)
    dataset_test.filter_max_answers(CFG["max_answers"], answer_set)
    

# For prediction head models
label2ans, ans2label = create_ans2label(dataset_train, dataset_validate, dataset_test)    
dataset_train.add_labels(ans2label)
dataset_validate.add_labels(ans2label)
dataset_test.add_labels(ans2label)

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

if "retrieval" in CFG and CFG["retrieval"]:
    if "k" in CFG:
        k = CFG["k"]
    else:
        k = 15
    if "use_additional_retrieval_data" in CFG and CFG["use_additional_retrieval_data"]:
        print(f"Using {k}-nn retrieval with additional synthetic data ...")
        dataset_train.create_retrieval_dataset(train_loader, MODEL_PREFIX, is_training_phase=args.train, retrieval_k=k, use_additional_data=True)
    else:
        print(f"Using {k}-nn retrieval with only training data ...")
        dataset_train.create_retrieval_dataset(train_loader, MODEL_PREFIX, is_training_phase=args.train, retrieval_k=k)
    retrieval_function = dataset_train.retrieve_closest_qa_pairs
else:
    retrieval_function = None

if CFG["use_prediction_head"]:
    if CFG["use_BAN"]:
        model = T5VisionModelPredictionHeadBAN(len(ans2label), vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"], glimpse=CFG["glimpse"], retrieval_function = retrieval_function).to(device)
    else:
        if "max_answers" in CFG and CFG["max_answers"]:
            model = T5VisionModelPredictionHead(CFG["max_answers"], vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"], retrieval_function = retrieval_function).to(device)
        else:
            model = T5VisionModelPredictionHead(len(ans2label), vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"], retrieval_function = retrieval_function).to(device)

else:
    if CFG["freeze"]:
        model = T5VisionModelFrozen(vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"], retrieval_function = retrieval_function).to(device)   
    else:
        model = T5VisionModel(vision_encoder=CFG["vision_encoder"], T5_version=CFG["T5_version"],use_image_info=use_image_info, vision_checkpoint=CFG["vision_checkpoint"], mapping_checkpoint=CFG["mapping_checkpoint"], retrieval_function = retrieval_function).to(device)



learning_rate = CFG["hyperparameters"]["learning_rate"]
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

if args.train or args.resume:
    if args.resume:
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_valid_loss = float("inf")
    best_epoch = 0
    longest_no_improvement_streak = 0
    train_info_path = os.path.join("logs", MODEL_PREFIX)
    os.makedirs(train_info_path, exist_ok=True)
    train_losses = []
    valid_losses = []
    parameter_updates = 0
    for epoch in range(CFG["hyperparameters"]["epochs"]):
        model.train()
        print(f"Starting epoch {epoch} ...")
        print(f"The learning rate is now {optimizer.param_groups[0]['lr']}")
        train_total = 0
        train_batch = 0
        total_ans = 0
        total_correct_ans = 0
        for batch in tqdm(train_loader):
            loss = model(batch)
            pred = model.predict(batch)

            if CFG["use_prediction_head"]:
                total_correct_ans += torch.sum(torch.eq(batch["label"].to(model.device), pred))
                total_ans += len(batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            parameter_updates += 1
            train_total += loss.item() * batch["image"].shape[0]
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
            longest_no_improvement_streak = 0
        else:
            longest_no_improvement_streak += 1

        train_losses.append((parameter_updates, train_total / len(train_loader.dataset)))
        valid_losses.append((parameter_updates, valid_loss))

        if longest_no_improvement_streak > 30:
           print(f"Loss didn't improve for {longest_no_improvement_streak - 1} epochs. Stopping training ...")
           break

    print(f"Writing training info to {train_info_path}")
    with open(os.path.join(train_info_path, "training_loss.txt"), "w") as f:
        f.write("parameter_updates,loss\n")
        for i in range(len(train_losses)):
            f.write(f"{train_losses[i][0]},{train_losses[i][1]}\n")
    
    with open(os.path.join(train_info_path, "validation_loss.txt"), "w") as f:
        f.write("parameter_updates,loss\n")
        for i in range(len(valid_losses)):
            f.write(f"{valid_losses[i][0]},{valid_losses[i][1]}\n")

# Test
if args.test:
    #MODEL_SAVE_PATH = "./models/model_SLAKE_with_vision.pt" # For OOD Testing
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=torch.device(device))
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

    string_match_correct = 0
    
    # For retrieval evaluation
    pred_in_retrieval = 0
    ground_truth_in_retrieval = 0
    full_retrieval_reliance_pred = 0
    full_retrieval_reliance_gt = 0
    ground_truth_consistency = []
    consistencies = [] 
    question_type_consistencies = []

    incorrect_ids = []
    correct_ids = []
    
    for batch in tqdm(test_loader):
        predicted_answers = model.predict(batch)

        if "retrieval" in CFG and CFG["retrieval"] and not CFG["use_prediction_head"]:

            retrieved_answers = train_loader.dataset.retrieve_closest_qa_pairs(batch, return_ans=True)
            # for i in range(len(retrieved_answers)):
            #     if retrieved_answers[i][0] != predicted_answers[i]:
            #         print(f"{retrieved_answers[i][0]} || {predicted_answers[i]} || {batch['answer'][i]}")
            # print(f"Retrieved answers: {retrieved_answers}")
            # print(f"Predicted answers: {predicted_answers}")
            # print(f"Ground Truth answers: {batch['answer']}")
            retrieved_answer_types = train_loader.dataset.retrieve_closest_qa_pairs(batch, return_ans_types=True)

            for i, pred_answer in enumerate(predicted_answers):
                
                
                answer_type = batch["question_type"][i]
                consistencies.append(sum([1 for x in retrieved_answers[i] if x == pred_answer.lower()]) / len(retrieved_answers[i]))
                ground_truth_consistency.append(sum([1 for x in retrieved_answers[i] if x == batch["answer"][i].lower()]) / len(retrieved_answers[i]))
                question_type_consistencies.append(sum([1 for x in retrieved_answer_types[i] if x == answer_type]) / len(retrieved_answer_types[i]))


                #print(retrieved_answers[i], max(set(retrieved_answers[i]), key=retrieved_answers[i].count), batch["answer"][i])
                if batch["answer"][i].lower() in retrieved_answers[i]:
                    ground_truth_in_retrieval += 1
                if pred_answer.lower() in retrieved_answers[i]:
                    pred_in_retrieval += 1
                if batch["answer"][i].lower() == max(set(retrieved_answers[i]), key=retrieved_answers[i].count):
                    full_retrieval_reliance_gt += 1
                if pred_answer.lower() == max(set(retrieved_answers[i]), key=retrieved_answers[i].count):
                    full_retrieval_reliance_pred += 1
        
        for i in range(len(predicted_answers)):
            if batch["task"][i] == "KG":
                print(f'{batch["question"][i]}|||{predicted_answers[i].lower()} ||| {batch["answer"][i]}')

            #print(test_loader.dataset.get_closest_label(batch["answer"][i].lower()), batch["label"][i], batch["answer"][i].lower(), predicted_answers[i].lower())
            string_matched = False
            if not CFG["use_prediction_head"]:
                if test_loader.dataset.get_closest_label(predicted_answers[i].lower()) == batch["label"][i].item():
                    string_match_correct += 1
                    if predicted_answers[i].lower() != batch["answer"][i].lower():
                        string_matched = True
                #print(string_match_correct)
            
            if CFG["use_prediction_head"]:
                is_correct = predicted_answers[i] == batch["label"][i]
            else:
                is_correct = predicted_answers[i].lower() == batch["answer"][i].lower() or string_matched
            
            if is_correct:
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
    for key in sorted(performance.keys()):
        val = performance[key]
        print(f"{key}: {100 * val:.1f}")
    print("=======OPEN VS CLOSED PERFORMANCE======")
    print(f"Open: {100 * open_correct/open_total:.1f}")
    print(f"Closed: {100 * closed_correct/closed_total:.1f}")
    print("===========OVERALL PERFORMANCE=========")
    print(f"Overall accuracy: {100 * sum(correct.values())/sum(total.values()):.1f}")


    if "retrieval" in CFG and CFG["retrieval"] and not CFG["use_prediction_head"]:
        print(f"Percentage of retrieved answers which == model prediction: {100 * sum(consistencies)/ len(consistencies):.1f}")
        print(f"Percentage of retrieved answers which == ground truth: {100 * sum(ground_truth_consistency)/ len(ground_truth_consistency):.1f}")
        print(f"Percentage of retrieved answers which have correct answer type: {100 * sum(question_type_consistencies)/ len(question_type_consistencies):.1f}")
        print(f"How often prediction is contained within set of retreieved answers: {100 * pred_in_retrieval / len(consistencies):.1f}")
        print(f"How often ground truth is contained within set of retrieved answers: {100 * ground_truth_in_retrieval / len(consistencies):.1f}")
        print(f"How often ground truth == most common retrieved answer: {100 * full_retrieval_reliance_gt / len(consistencies):.1f}")
        print(f"How often prediction == most common retrieved answer: {100 * full_retrieval_reliance_pred / len(consistencies):.1f}")


        #print([batch['answer'][i] + "|||" + predicted_answers[i] for i in range(len(predicted_answers))])
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "incorrect_ids.txt"), "w") as f:
        for qid in incorrect_ids:
            f.write(qid + "\n")
    
    with open(os.path.join("logs", "correct_ids.txt"), "w") as f:
        for qid in correct_ids:
            f.write(qid + "\n")

    with open(os.path.join("logs", MODEL_PREFIX + "performance.txt"), "w") as f:
        for key in  sorted(performance.keys()):
            val = performance[key]
            f.write(f"{100 * val:.1f}\n")
        f.write(f"Open,{(open_correct/open_total):.4f}\n")
        f.write(f"Closed: {(closed_correct/closed_total):.4f}\n")
        f.write(f"Overall,{(sum(correct.values())/sum(total.values())):.4f}")

if args.eval:
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=torch.device(device))
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
            if batch["task"][0] == "KG":
                visualize_attn_weights(model, batch, attn_type = "encoder_attentions")
                print(f"Finished image {i} out of {num_lines}")