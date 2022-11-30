from wsgiref import validate
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
import torch
from copy import deepcopy
from PIL import Image

def get_model_prefix(CFG):
    data_name = CFG["dataset"]
    use_image_info = bool(CFG["use_image_info"])

    MODEL_PREFIX = f"model_{data_name}"
    if use_image_info:
        MODEL_PREFIX += "_with_vision"
    else:
        MODEL_PREFIX += "_no_vision"

    if CFG["vision_checkpoint"]:
        MODEL_PREFIX += "_with_pretrained_checkpoint"
    else:
        MODEL_PREFIX += "_no_pretrained_checkpoint"

    if "fewshot_training_tasks" in CFG and CFG["fewshot_training_tasks"]["enabled"]:
        MODEL_PREFIX += "_fewshot"

    if "mapping_checkpoint" in CFG and CFG["mapping_checkpoint"]:
        MODEL_PREFIX += "_with_mapping"

    if CFG["use_prediction_head"]:
        if CFG["use_BAN"]:
            MODEL_PREFIX += "_pred_head_BAN"
        else:
            MODEL_PREFIX += "_pred_head"

    if "freeze" in CFG and CFG["freeze"]:
        MODEL_PREFIX += "_freeze" 

    if "retrieval" in CFG and CFG["retrieval"]:
        MODEL_PREFIX += "_retrieval"

    if "RN" in CFG["vision_encoder"]:
        MODEL_PREFIX += "_resnet"

    return MODEL_PREFIX

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def create_ans2label(dataset_train, dataset_validate, dataset_test):
    samples = deepcopy(dataset_train.entries)
    samples.extend(dataset_validate.entries)
    samples.extend(dataset_test.entries)
    possible_answers = sorted(set([sample['answer'].lower() for sample in samples]))
    ans_to_label = {}
    label_to_ans = {}
    for i in range(len(possible_answers)):
        label_to_ans[i] = possible_answers[i]
        ans_to_label[possible_answers[i]] = i


    return label_to_ans, ans_to_label

def get_validation_loss(model, validate_loader):
    print("Calculating Validation Loss ...")
    model.eval()
    total = 0
    with torch.no_grad():
        for batch in tqdm(validate_loader):
            loss = model(batch)
            # total_correct_ans += torch.sum(torch.eq(batch["labels"].to(model.device), pred))
            # total_ans += len(batch["labels"])
            
            total += loss.item() * batch["image"].shape[0]
        #print(f"Valid acc is: {total_correct_ans / total_ans}")
        return total / len(validate_loader.dataset)

# Weights are tuple of length n_layers
# Each entry of tuple is of shape (batch_sz, n_heads, seq_len, seq_len)
def visualize_attn_weights(model, batch, attn_type="encoder_attentions"):

    combined_embedding, attention_mask, encoding = model.prepare_input(batch)
    n_image_tokens = 50
    grid_size = int(n_image_tokens ** 0.5)

    n_padding = combined_embedding.shape[1] - sum(attention_mask[0])
    n_padding = int(n_padding.item())

    final_tokens_X = model.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
    final_tokens_X = ["ITK"] * 50 + final_tokens_X

    output_sequences = model.T5_model.generate(
        inputs_embeds = combined_embedding,
        attention_mask = attention_mask,
        do_sample=False,  # disable sampling to test if batching affects output
        max_new_tokens=20
        )

    if attn_type == "encoder_attentions":
        final_tokens_Y = final_tokens_X
    elif attn_type == "cross_attentions":
        final_tokens_Y = model.tokenizer.convert_ids_to_tokens(output_sequences[0])
        #final_tokens_Y = ["ITK"] * 50 + final_tokens_Y
    
    predicted_answers = model.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    output = model.T5_model(inputs_embeds = combined_embedding, labels=output_sequences, use_cache=False, output_attentions=True, return_dict=True)
    
    weights = output[attn_type]

    
    
    x_ticks = np.linspace(0, weights[0].shape[3] - 1, weights[0].shape[3])
    y_ticks = np.linspace(0, weights[0].shape[2] - 1, weights[0].shape[2])
    
    
    print(len(final_tokens_X), len(x_ticks))
    # 6 layers, each layer has 8 attention heads
    

    n_layers = len(weights)
    n_heads = weights[0].shape[1]
    

    assert len(final_tokens_X) == len(x_ticks)
    assert len(final_tokens_Y) == len(y_ticks)

    original_image = Image.open(batch["path_to_image"][0])
    original_image = original_image.resize((224,224))
    image_x_ticks = np.linspace(0, original_image.width, grid_size + 1)
    image_y_ticks = np.linspace(0, original_image.height, grid_size + 1)
    grid_x_length = image_x_ticks[1] - image_x_ticks[0]
    grid_y_length = image_y_ticks[1] - image_y_ticks[0]

    for i in range(n_layers):
        for j in range(n_heads):
            fig, ax = plt.subplots(1, len(final_tokens_Y) + 1, figsize=(min(len(final_tokens_Y) * 2, 650),20))
            #if i == 3 and j == 0:

            ax[0].imshow(weights[i][0,j,:,:].detach().cpu().numpy(), vmin=0, vmax=1)

            ax[0].set_title(batch["question"][0])
            ax[0].set_xlabel(f"Correct Answer: {batch['answer'][0]} \n Predicted Answer: {predicted_answers[0]}")
            ax[0].set_xticks(x_ticks)
            ax[0].set_yticks(y_ticks)
            ax[0].set_xticklabels(final_tokens_X)
            ax[0].set_yticklabels(final_tokens_Y)
            ax[0].tick_params(axis='x', labelrotation = 90)
            #ax[0].grid()

            for k in range(len(final_tokens_Y)):
                ax[k + 1].set_title(final_tokens_Y[k])
                ax[k + 1].imshow(original_image)
                ax[k + 1].set_xticks(image_x_ticks)
                ax[k + 1].set_yticks(image_y_ticks)

                if attn_type == "encoder_attentions":
                    alphas = weights[i][0,j,1:51,k].detach().cpu().numpy()
                elif attn_type == "cross_attentions":
                    alphas = weights[i][0,j,k,1:51].detach().cpu().numpy()
                    
                for l in range(grid_size):
                    for m in range(grid_size):

                        rect = patches.Rectangle((image_x_ticks[m], image_y_ticks[l]), grid_x_length, grid_y_length, linewidth=1, fill=True, facecolor="red", alpha=alphas[grid_size * l + m])
                        # Add the patch to the Axes
                        ax[k + 1].add_patch(rect)
                ax[k + 1].grid()



            os.makedirs(os.path.join("figures", batch["question_id"][0], f"head{j}"), exist_ok=True)
            plt.savefig(os.path.join("figures", batch["question_id"][0], f"head{j}", f"attention{i}.png"))
            plt.close()
        
