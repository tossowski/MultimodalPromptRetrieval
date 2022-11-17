import os
from PIL import Image
from matplotlib import pyplot as plt
import random
from random import sample
import pandas as pd
from ROCO import ROCOFeatureDataset
from torch.utils.data import DataLoader

random.seed(88)

ORGAN_SYSTEMS = ['Brain', 'Chest', 'Cardiovascular System', 'Respiratory System', 'Gastrointestinal System', 'Cardiopulmonary System'] # GI CNS?
ORGANS = ['Heart', 'Lungs', 'Lung', 'Liver', 'Breasts']

organ_system_question_open_templates = ['What system is this pathology in?', 'What organ system is pictured?', 'What organ system is evaluated primarily?', 'What is the organ system visualized?', 'What organ system is displayed?']
organ_system_question_closed_templates = ['Is this an image of the {}?', 'Is this a study of the {}?', 'Is this the {}?', 'Is the {} shown?']
organ_question_open_templates = ['What part of the body is being imaged?', 'What is the organ principally shown in this image?']

PATH_TO_ROCO_DATA = "/data/ossowski/roco-dataset/data/train/radiology"

caption_path = os.path.join(PATH_TO_ROCO_DATA, "captions.txt")
semtype_path = os.path.join(PATH_TO_ROCO_DATA, "semtypes.txt")
keywords_path = os.path.join(PATH_TO_ROCO_DATA, "keywords.txt")
images_path = os.path.join(PATH_TO_ROCO_DATA, "images")

captions = {}
with open(caption_path, "r") as f:
    for line in f:
        roco_id, caption = line.split("\t", 1)
        captions[roco_id] = caption

keywords = {}
with open(keywords_path, "r") as f:
    for line in f:
        roco_id, k = line.split("\t", 1)
        keywords[roco_id] = [x.lower() for x in k.split("\t")]
        #print(keywords[roco_id])

col_names = ['q_type', 'image_id', 'question', 'answer', 'caption', 'question_type']
row_data = []

for roco_id in keywords:
    if not os.path.exists(os.path.join(images_path, roco_id + ".jpg")):
        print(f"{os.path.join(images_path, roco_id + '.jpg')} doesn't exist!!! Skipping ...")
        continue
    for system in ORGAN_SYSTEMS:
        org_sys = system.split(" ")[0].lower()
        
        if org_sys in keywords[roco_id]:
            question = sample(organ_system_question_open_templates, 1)[0]
            answer = system
            row_data.append(["Organ", roco_id + ".jpg", question, answer, captions[roco_id], 'Open'])

    for system in ORGANS:
        org_sys = system.split(" ")[0].lower()
        
        if org_sys in keywords[roco_id]:
            question = sample(organ_question_open_templates, 1)[0]
            answer = system
            row_data.append(["Organ", roco_id + ".jpg", question, answer, captions[roco_id], 'Open'])
    #if "gastrointestinal" in keywords[roco_id]:
    #    print(keywords[roco_id])

df = pd.DataFrame(row_data)
df.columns = col_names
df.to_csv('/data/ossowski/ROCO/train.csv', index=False)

# Testing to see if it worked ...
data = ROCOFeatureDataset("train", "/data/ossowski/ROCO", device="cpu")
train_loader = DataLoader(data, 1, shuffle=True, num_workers=2)

for batch in train_loader:
    print(batch["question"], batch["answer"])
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.title(row_data[idx][2] + "\n" + row_data[idx][3])
# plt.savefig("generate_output.png")