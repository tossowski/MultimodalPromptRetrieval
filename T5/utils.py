from tqdm import tqdm

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