import os


from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from ROCO import ROCOFeatureDataset
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

MODEL_SAVE_FOLDER = "models"
MODEL_SAVE_PATH = "models/crossmodal_mapping.pt"

class CrossModalMapping(nn.Module):
    def __init__(self, in_dim = 512, out_dim = 512):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()


    def forward(self, batch, device="cuda"):
        mapped_image_feats =  self.linear_relu_stack(batch["clip_image_features"].to(device))
        logit_scale = self.logit_scale.exp()
        # logits_per_image_clip = logit_scale * batch["clip_image_features"].to(device) @ batch["clip_text_features"].to(device).t()
        logits_per_image_t5 = logit_scale * mapped_image_feats @ batch["t5_text_features"].to(device).t()
        return logits_per_image_t5

    def forward2(self, batch):
        return self.linear_relu_stack(batch["clip_text_features"])

    
    def loss2(self, mapped_text_feats, actual_text_feats):
        #print(torch.norm(mapped_text_feats - actual_text_feats))
        #print(self.mse_loss(mapped_text_feats, actual_text_feats))
        return self.mse_loss(mapped_text_feats, actual_text_feats)


def visualize_mapping(image_vecs, text_vecs, text_vecs_t5, image_vecs_t5, save_path="mapping.png"):
    image_data = np.stack(image_vecs, axis=0)
    text_data = np.stack(text_vecs, axis=0)
    t5_data = np.stack(text_vecs_t5, axis=0)
    t5_images = np.stack(image_vecs_t5, axis=0)
    data = np.concatenate((image_data, text_data, t5_data, t5_images), axis=0)

    scaler = StandardScaler()
    scaler.fit(data)
    data=scaler.transform(data)    


    pca = PCA()
    fitted_data = pca.fit_transform(data)
    x_new_image = fitted_data[:len(image_data)]
    x_new_text = fitted_data[len(image_data): 2 * len(image_data)]
    x_new_t5_text = fitted_data[2 * len(image_data):3 * len(image_data)]
    x_new_t5_images = fitted_data[3 * len(image_data):]

    # np.random.seed(1)
    # x_new = TSNE(n_components=2).fit_transform(data)
    print(len(x_new_image), len(x_new_text), len(x_new_t5_text), len(x_new_t5_images))

    #print(pca.explained_variance_ratio_)

    fig = plt.figure()
    scatter_images = plt.scatter(x_new_image[:,0], x_new_image[:,1], label="image_features")
    scatter_text = plt.scatter(x_new_text[:,0], x_new_text[:,1], label="text_features")
    scatter_T5 = plt.scatter(x_new_t5_text[:,0], x_new_t5_text[:,1], label="t5_text_features")
    scatter_T5_image = plt.scatter(x_new_t5_images[:,0], x_new_t5_images[:,1], label="t5_image_features")


    plt.title("CLIP and T5 Image and Text Features on ROCO Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Feature Type")
    plt.savefig(save_path)



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset_train = ROCOFeatureDataset("train", clip_type="standard")
    # dataset_validate = ROCOFeatureDataset("validation", clip_type="standard")
    # dataset_test = ROCOFeatureDataset("test", clip_type="standard")

    dataset_train = ROCOFeatureDataset("train")
    dataset_validate = ROCOFeatureDataset("validation")
    dataset_test = ROCOFeatureDataset("test")

    train_loader = DataLoader(dataset_train, 16, shuffle=True, num_workers=2)
    validate_loader = DataLoader(dataset_validate, 16, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset_test, 16, shuffle=True, num_workers=2)

    model = CrossModalMapping(512, 512).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


    print("Calculating validation loss ...")
    total = 0
    image_vecs = []
    text_vecs = []
    text_vecs_t5 = []
    mapped_vecs = []
    for batch in tqdm(validate_loader):
        logits_per_image = model(batch, device=device)
        logits_per_text = logits_per_image.t()
        ground_truth = torch.arange(len(batch["clip_text_features"]),dtype=torch.long,device=device)
        total_loss = (model.loss_img(logits_per_image,ground_truth) + model.loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()


        total += total_loss.item()

        image_vecs.append(batch["clip_image_features"].detach().cpu().numpy()[0])
        text_vecs.append(batch["clip_text_features"].detach().cpu().numpy()[0])
        text_vecs_t5.append(batch["t5_text_features"].detach().cpu().numpy()[0])
        #mapped_vecs.append(mapped_feats.detach().cpu().numpy()[0])
        mapped_image_feats = model.linear_relu_stack(batch["clip_image_features"].to(device)).detach().cpu().numpy()[0]
        mapped_vecs.append(mapped_image_feats)
    val_loss = total / len(validate_loader.dataset)
    scheduler.step(val_loss)
    print(f"Average validation batch loss is {val_loss}")
    visualize_mapping(image_vecs, text_vecs, text_vecs_t5, mapped_vecs, save_path=f"before.png")

    best_valid_loss = float("inf")
    best_epoch = 0
    for epoch in range(100):
        #model.train()
        print(f"Starting epoch {epoch} ...")

        for batch in tqdm(train_loader):
            logits_per_image = model(batch, device=device)
            logits_per_text = logits_per_image.t()
            ground_truth = torch.arange(len(batch["clip_text_features"]),dtype=torch.long,device=device)
            total_loss = (model.loss_img(logits_per_image,ground_truth) + model.loss_txt(logits_per_text,ground_truth))/2
            optimizer.zero_grad()
            
            total_loss.backward()
            optimizer.step()



        print("Calculating validation loss ...")
        total = 0
        num_batch = 0
        image_vecs = []
        text_vecs = []
        text_vecs_t5 = []
        mapped_vecs = []
        for batch in tqdm(validate_loader):
            logits_per_image = model(batch, device=device)
            logits_per_text = logits_per_image.t()
            ground_truth = torch.arange(len(batch["clip_text_features"]),dtype=torch.long,device=device)
            total_loss = (model.loss_img(logits_per_image,ground_truth) + model.loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()

            #mapped_feats = model.forward2(batch)
            
            #loss = model.loss2(mapped_feats, batch["t5_text_features"])

            total += total_loss.item()

            image_vecs.append(batch["clip_image_features"].detach().cpu().numpy()[0])
            text_vecs.append(batch["clip_text_features"].detach().cpu().numpy()[0])
            text_vecs_t5.append(batch["t5_text_features"].detach().cpu().numpy()[0])
            #mapped_vecs.append(mapped_feats.detach().cpu().numpy()[0])

            mapped_image_feats = model.linear_relu_stack(batch["clip_image_features"].to(device)).detach().cpu().numpy()[0]
            mapped_vecs.append(mapped_image_feats)

        val_loss = total / len(validate_loader.dataset)
        scheduler.step(val_loss)
        print(f"Average validation batch loss is {val_loss}")


        if val_loss < best_valid_loss:
            print(f"Saving model to {MODEL_SAVE_PATH} ...")
            visualize_mapping(image_vecs, text_vecs, text_vecs_t5, mapped_vecs, save_path=f"epoch{epoch}.png")
            os.makedirs(MODEL_SAVE_FOLDER, exist_ok = True)
            torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, MODEL_SAVE_PATH)
            best_valid_loss = val_loss
            best_epoch = epoch
        # print(f"Validation Loss: {valid_loss} | Best Validation Loss: {best_valid_loss} at epoch {best_epoch}")
        # if valid_loss < best_valid_loss:
        #     print(f"Saving model to {MODEL_SAVE_PATH} ...")
        #     os.makedirs(MODEL_SAVE_FOLDER, exist_ok = True)
        #     torch.save({'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict()}, MODEL_SAVE_PATH)
        #     best_valid_loss = valid_loss
        #     best_epoch = epoch

    # Evaluating image-text and text-image matching

    # image-text
    image_vecs = []
    text_vecs = []
    text_vecs_t5 = []
    mapped_vecs = []
    for batch in tqdm(test_loader):
        image_vecs.extend(batch["clip_image_features"].detach().cpu().numpy())
        text_vecs.extend(batch["clip_text_features"].detach().cpu().numpy())
        text_vecs_t5.extend(batch["t5_text_features"].detach().cpu().numpy())
        #mapped_vecs.append(mapped_feats.detach().cpu().numpy()[0])
        mapped_image_feats = model.linear_relu_stack(batch["clip_image_features"].to(device)).detach().cpu().numpy()
        mapped_vecs.extend(mapped_image_feats)
    image_vecs = torch.FloatTensor(np.array(image_vecs)).to(device)
    text_vecs = torch.FloatTensor(np.array(text_vecs)).to(device)
    text_vecs_t5 = torch.FloatTensor(np.array(text_vecs_t5)).to(device)
    mapped_vecs = torch.FloatTensor(np.array(mapped_vecs)).to(device)

    # image-text
    similarity = (100.0 * image_vecs @ text_vecs.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    correct = 0
    total = 0
    for idx, row in enumerate(similarity):
        values, indices =  row.topk(5)
        if idx in indices:
            correct += 1
        total += 1
    print(f"CLIP image to text: {correct / total}")

    similarity = (100.0 * text_vecs.T @ image_vecs).softmax(dim=-1)
    correct = 0
    total = 0
    for idx, row in enumerate(similarity):
        values, indices =  row.topk(5)
        if idx in indices:
            correct += 1
        total += 1
    print(f"CLIP text to image: {correct / total}")


    similarity = (100.0 * mapped_vecs @ text_vecs_t5.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    correct = 0
    total = 0
    for idx, row in enumerate(similarity):
        values, indices =  row.topk(5)
        if idx in indices:
            correct += 1
        total += 1
    print(f"T5 image to text: {correct / total}")

    similarity = (100.0 * text_vecs_t5 @ image_vecs.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)
    correct = 0
    total = 0
    for idx, row in enumerate(similarity):
        values, indices =  row.topk(5)
        if idx in indices:
            correct += 1
        total += 1
    print(f"T5 text to image: {correct / total}")
    #print(indices)
    #from scipy.spatial import distance

    # distances = distance.cdist([target], vectors, "cosine")[0]
    # min_index = np.argmin(distances)
    # min_distance = distances[min_index]
    # max_similarity = 1 - min_distance
    
    # distances = distance.cdist(image_vecs, text_vecs, "cosine")
    # dist_sorted = np.argsort(distances, axis=1)
    # print(dist_sorted)
   # print(image.shape)