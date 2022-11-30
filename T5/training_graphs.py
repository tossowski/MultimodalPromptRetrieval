from matplotlib import pyplot as plt
import os
import pandas as pd

checkpoint_name_to_label = {
    "model_SLAKE_with_vision_no_pretrained_checkpoint": "default",
    "model_SLAKE_with_vision_with_pretrained_checkpoint": "vision checkpoint",
    "model_SLAKE_with_vision_with_pretrained_checkpoint_pred_head": "prediction head",
    "model_SLAKE_with_vision_with_pretrained_checkpoint_pred_head_BAN": "prediction head + BAN",
    "model_VQA_RAD_with_vision_no_pretrained_checkpoint": "default",
    "model_VQA_RAD_with_vision_with_pretrained_checkpoint": "vision checkpoint",
    "model_VQA_RAD_with_vision_with_pretrained_checkpoint_pred_head": "prediction head",
    "model_VQA_RAD_with_vision_with_pretrained_checkpoint_pred_head_BAN": "prediction head + BAN"
}

train_data = {}
validate_data = {}

for fname in os.listdir("logs"):
    if os.path.isdir(os.path.join("logs", fname)):
        print(fname)
        train_data[fname] = {}
        train = pd.read_csv(os.path.join("logs", fname, "training_loss.txt"))
        validation = pd.read_csv(os.path.join("logs", fname, "validation_loss.txt"))
        best_epoch = validation["loss"].idxmin()
        train_data[fname]["best_epoch"] = best_epoch
        train_data[fname]["train"] = list(train["loss"])
        train_data[fname]["validation"] = list(validation["loss"])
        
        train_data[fname]["updates"] = list(validation["parameter_updates"])
        print(len(train_data[fname]["updates"]), len(train_data[fname]["train"]))

for model in train_data:
    label = checkpoint_name_to_label[model]

    
    if "SLAKE" in model:
        plt.plot(train_data[model]["updates"], train_data[model]["validation"], label=label)

plt.title("Validation Loss as a Function of Parameter Updates")
plt.xlabel("Number of Parameter Updates")
plt.ylabel("Average Cross Entropy Loss")
plt.legend()
plt.savefig("train_fig.png")



