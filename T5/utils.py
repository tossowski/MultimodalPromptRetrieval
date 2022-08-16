from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import numpy as np

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
def visualize_attn_weights(model, batch):

    combined_embedding, attention_mask, encoding = model.prepare_input(batch)
    n_image_tokens = combined_embedding.shape[1] - len(encoding.input_ids[0])
    n_padding = combined_embedding.shape[1] - sum(attention_mask[0])
    n_padding = int(n_padding.item())

    # - sum(attention_mask[0]))
    image_tokens = [f"image_token_{i}" for i in range(n_image_tokens)]
    normal_tokens = [x for x in model.tokenizer.convert_ids_to_tokens(encoding.input_ids[0]) if x != model.tokenizer.pad_token]
    
    padding_tokens = [model.tokenizer.pad_token for i in range(n_padding)]

    final_tokens = normal_tokens + image_tokens + padding_tokens
    
    output_sequences = model.T5_model.generate(
        inputs_embeds = combined_embedding,
        attention_mask = attention_mask,
        do_sample=False,  # disable sampling to test if batching affects output
        max_new_tokens=20
        )

    output = model.T5_model(inputs_embeds = combined_embedding, labels=output_sequences, use_cache=False, output_attentions=True, return_dict=True)
    ticks = np.linspace(0, len(final_tokens) - 1, len(final_tokens))

    # 6 layers, each layer has 8 attention heads
    weights = output['encoder_attentions']
    #print(output['encoder_attentions'][0].shape, combined_embedding.shape)
    #return predicted_answers, output['encoder_attentions']
    n_layers = len(weights)
    n_heads = weights[0].shape[1]
    fig, ax = plt.subplots(1, 1, figsize=(20,20))

    assert len(final_tokens) == combined_embedding.shape[1]

    for i in range(n_layers):
        for j in range(n_heads):
            plt.imshow(weights[i][0,j,:,:].detach().cpu().numpy())
            plt.title(batch["question"][0])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(final_tokens)
            ax.set_yticklabels(final_tokens)
            plt.xticks(rotation = 90)
            os.makedirs(f"figures/head{j}", exist_ok=True)
            plt.savefig(f"figures/head{j}/attention{i}.png")
