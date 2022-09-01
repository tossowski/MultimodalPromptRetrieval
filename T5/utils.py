from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import numpy as np
from PIL import Image

def get_validation_loss(model, validate_loader):
    print("Calculating Validation Loss ...")
    model.eval()
    n_batches = 0
    total = 0
    for batch in tqdm(validate_loader):
        loss = model(batch)
        total += loss.item()
        n_batches += 1
    return total / n_batches

# Weights are tuple of length n_layers
# Each entry of tuple is of shape (batch_sz, n_heads, seq_len, seq_len)
def visualize_attn_weights(model, batch, attn_type="encoder_attentions"):

    combined_embedding, attention_mask, encoding = model.prepare_input(batch)
    n_image_tokens = 50
    grid_size = int(n_image_tokens ** 0.5)

    n_padding = combined_embedding.shape[1] - sum(attention_mask[0])
    n_padding = int(n_padding.item())

    final_tokens_X = model.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
    
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
    
    predicted_answers = model.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    output = model.T5_model(inputs_embeds = combined_embedding, labels=output_sequences, use_cache=False, output_attentions=True, return_dict=True)
    
    weights = output[attn_type]

    x_ticks = np.linspace(0, weights[0].shape[3] - 1, weights[0].shape[3])
    y_ticks = np.linspace(0, weights[0].shape[2] - 1, weights[0].shape[2])
    
    
    # 6 layers, each layer has 8 attention heads
    

    n_layers = len(weights)
    n_heads = weights[0].shape[1]
    fig, ax = plt.subplots(1, 2, figsize=(20,20))

    assert len(final_tokens_X) == len(x_ticks)
    assert len(final_tokens_Y) == len(y_ticks)

    original_image = Image.open(batch["path_to_image"][0])
    image_x_ticks = np.linspace(0, original_image.width, grid_size + 1)
    image_y_ticks = np.linspace(0, original_image.height, grid_size + 1)

    for i in range(n_layers):
        for j in range(n_heads):
            ax[0].imshow(weights[i][0,j,:,:].detach().cpu().numpy())
            ax[0].set_title(batch["question"][0])
            ax[0].set_xlabel(f"Correct Answer: {batch['answer'][0]} \n Predicted Answer: {predicted_answers[0]}")
            ax[0].set_xticks(x_ticks)
            ax[0].set_yticks(y_ticks)
            ax[0].set_xticklabels(final_tokens_X)
            ax[0].set_yticklabels(final_tokens_Y)
            ax[0].tick_params(axis='x', labelrotation = 90)
            #ax[0].grid()
            ax[1].imshow(original_image)
            ax[1].set_xticks(image_x_ticks)
            ax[1].set_yticks(image_y_ticks)
            ax[1].grid()
            
            
            os.makedirs(os.path.join("figures", batch["question_id"][0], f"head{j}"), exist_ok=True)
            plt.savefig(os.path.join("figures", batch["question_id"][0], f"head{j}", f"attention{i}.png"))
            
        
