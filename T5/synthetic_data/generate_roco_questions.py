import os
import torch
import pandas as pd
import sys
import clip
import pickle

sys.path.insert(0, os.path.abspath(".."))
from random import sample
from tqdm import tqdm
from ROCO import ROCOFeatureDataset
from torch.utils.data import DataLoader
from question_category import QuestionCategoryBucket

if __name__ == "__main__":

    seed = 88

    # ORGAN KEYWORDS
    ORGAN_SYSTEMS = ['Brain', 'Chest', 'Cardiovascular System', 'Respiratory System', 'Gastrointestinal System', 'Cardiopulmonary System'] # GI CNS?
    ORGANS = ['Heart', 'Lungs', 'Lung', 'Liver', 'Breasts']

    # ORGAN TEMPLATES
    organ_system_question_open_templates = ['What system is this pathology in?', 'What organ system is pictured?', 'What organ system is evaluated primarily?', 'What is the organ system visualized?', 'What organ system is displayed?']
    organ_system_question_closed_templates = ['Is this an image of the {}?', 'Is this a study of the {}?', 'Is this the {}?', 'Is the {} shown?']
    organ_question_open_templates = ['What part of the body is being imaged?', 'What is the organ principally shown in this image?']

    # MODALITY KEYWORDS
    MODALITIES = ['MRI', 'T1', 'T2', 'CT', 'xray', 'ultrasound', 'flair']

    # MODALITY TEMPLATES
    modality_question_open_templates = ["What is the modality by which the image was taken?", "What kind of scan is this?", "How was this image taken", "What type of imaging modality is seen in this image?", "What is the modality used?", "What imaging method was used?", "What modality is this?"]
    modality_question_closed_templates = ["Is this a {}?", "Is the image an {}?"]

    # PLANE KEYWORDS
    PLANES = ['axial', 'coronal', 'supratentorial', 'posteroanterior']

    # PLANE TEMPLATES
    plane_question_open_templates = ["In what plane is this image oriented?", "Which plane is this image taken?", "What is the name of this image's plane?", "How is the image oriented?", "What image plane is this?", "What plane are we in?"]
    plane_question_closed_templates = ["Is this a {} plane?", "Is this a {} image?", "Is this a {} section?", "Was this iamge taken in {} format?"]

    # PRESENCE KEYWORDS
    PRESENCE = ['pneumothorax', 'fracture', 'hernia', 'edema', 'hematoma', 'cyst', 'hemorrhage', 'lymphadenopathy', 'pneumoperitoneum']

    # PRESENCE TEMPLATES
    presence_question_closed_templates = ["Is there evidence of a {}?", "Is there a {}", "Is a {} present?"]

    PATH_TO_ROCO_DATA = "/data/ossowski/roco-dataset/data/train/radiology"

    caption_path = os.path.join(PATH_TO_ROCO_DATA, "captions.txt")
    semtype_path = os.path.join(PATH_TO_ROCO_DATA, "semtypes.txt")
    keywords_path = os.path.join(PATH_TO_ROCO_DATA, "keywords.txt")
    images_path = os.path.join(PATH_TO_ROCO_DATA, "images")

    # Create category buckets
    ORGAN_SYSTEM_OPEN = QuestionCategoryBucket("Organ", ORGAN_SYSTEMS, organ_system_question_open_templates, "open", seed)
    ORGAN_SYSTEM_CLOSED = QuestionCategoryBucket("Organ", ORGAN_SYSTEMS, organ_system_question_closed_templates, "closed", seed)
    ORGAN_OPEN = QuestionCategoryBucket("Organ", ORGANS, organ_question_open_templates, "open", seed)
    MODALITY_OPEN = QuestionCategoryBucket("Modality", MODALITIES, modality_question_open_templates, "open", seed)
    MODALITY_CLOSED = QuestionCategoryBucket("Modality", MODALITIES, modality_question_closed_templates, "closed", seed)
    PLANE_OPEN = QuestionCategoryBucket("Plane", PLANES, plane_question_open_templates, "open", seed)
    PLANE_CLOSED = QuestionCategoryBucket("Plane", PLANES, plane_question_closed_templates, "closed", seed)
    PRESENCE_CLOSED = QuestionCategoryBucket("Presence", PRESENCE, presence_question_closed_templates, "closed", seed)


    question_buckets = [ORGAN_SYSTEM_OPEN, ORGAN_SYSTEM_CLOSED, ORGAN_OPEN, MODALITY_OPEN, MODALITY_CLOSED]

    captions = {}
    with open(caption_path, "r") as f:
        for line in f:
            roco_id, caption = line.split("\t", 1)
            captions[roco_id] = caption

    keywords = {}
    with open(keywords_path, "r") as f:
        for line in f:
            roco_id, k = line.split("\t", 1)
            keywords[roco_id] = [x.lower() for x in k.split("\t")][1:]
            # if 'pneumoperitoneum' in keywords[roco_id]:
            #     print("asdfadsfadfadfasdf")

    col_names = ['q_type', 'image_id', 'question', 'answer', 'question_type']
    row_data = []

    for roco_id in keywords:
        if not os.path.exists(os.path.join(images_path, roco_id + ".jpg")):
            print(f"{os.path.join(images_path, roco_id + '.jpg')} doesn't exist!!! Skipping ...")
            continue

        for q_bucket in question_buckets:
            out = q_bucket.get_question(keywords[roco_id])
            if out == None:
                continue

            questions, answers = out
            if questions and answers:
                for i in range(len(questions)):
                    question = questions[i]
                    answer = answers[i]
                    row_data.append([q_bucket.q_category, roco_id + ".jpg", question, answer, q_bucket.q_type])
        #if "gastrointestinal" in keywords[roco_id]:
        #    print(keywords[roco_id])

    df = pd.DataFrame(row_data)
    df.columns = col_names
    df.to_csv('/data/ossowski/ROCO/train.csv', index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    data = ROCOFeatureDataset("train", "/data/ossowski/ROCO", device=device)
    train_loader = DataLoader(data, 1, shuffle=True, num_workers=2)

    os.makedirs(os.path.join("cache", "ROCOFeatureDataset"), exist_ok=True)
    embedding_path = os.path.join("cache", "ROCOFeatureDataset", "embedding.pt")
    answer_path = os.path.join("cache", "ROCOFeatureDataset", "answers.pkl")

    print(f"Creating qa pairs in {os.path.join('cache', 'ROCOFeatureDataset')} ...")
    all_embeddings = []
    all_answers = []
    for batch in tqdm(train_loader):
        image_encoding = clip_model.encode_image(batch["image"].to(device))
        text_encoding = clip_model.encode_text(clip.tokenize(batch["question"]).to(device))
        answers = batch["answer"]
        all_embeddings.append(torch.cat([image_encoding,text_encoding], axis=1).detach())
        all_answers.extend(answers)

    retrieval_embeddings = torch.cat(all_embeddings, axis=0).float()
    retrieval_answers = all_answers

    torch.save(retrieval_embeddings, embedding_path)
    with open(answer_path, 'wb') as f:
        pickle.dump(all_answers, f)


    # Testing to see if it worked ...
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = ROCOFeatureDataset("train", "/data/ossowski/ROCO", device="cpu")
    train_loader = DataLoader(data, 1, shuffle=True, num_workers=2)

    for batch in train_loader:
        print(batch["question"], batch["answer"])
    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    # plt.title(row_data[idx][2] + "\n" + row_data[idx][3])
    # plt.savefig("generate_output.png")