import os
import shutil
import torch
import pandas as pd
import sys
import clip
import pickle


sys.path.insert(0, os.path.abspath(os.getcwd()))
from random import sample
from tqdm import tqdm
from torch.utils.data import DataLoader
from question_category import QuestionCategoryBucket
from question_category_specific import SpecificQuestionCategoryBucket

if __name__ == "__main__":

    PATH_TO_ROCO_DATA = sys.argv[1]
    SAVE_PATH = sys.argv[2] # Should be the same as datafolder in your config

    seed = 88

    # ORGAN KEYWORDS
    ORGAN_SYSTEMS = ['Brain', 'Chest', 'Cardiovascular System', 'Respiratory System', 'Gastrointestinal System', 'Cardiopulmonary System'] # GI CNS?
    #ORGANS = ['Abdomen', 'Brain', 'Chest', 'Neck', 'Head', 'Heart', 'Lungs', 'Lung', 'Liver', 'Breasts', 'Kidney', 'Spleen']
    #ORGANS = ['Heart', 'Lung', 'Liver', 'Kidney']
    ORGANS = ['Heart', 'Lungs', 'Lung', 'Liver', 'Breasts']

    # ORGAN TEMPLATES
    organ_system_question_open_templates = ['What system is this pathology in?', 'What organ system is pictured?', 'What organ system is evaluated primarily?', 'What is the organ system visualized?', 'What organ system is displayed?']
    organ_system_question_closed_templates = ['Is this an image of the {}?', 'Is this a study of the {}?', 'Is this the {}?', 'Is the {} shown?']
    organ_question_open_templates = ['What part of the body is being imaged?', 'What is the organ principally shown in this image?']
    organ_question_closed_templates = ['Does the picture contain {}?', 'Is this a study of the {}?', 'Does the {} appear in this image?']

    # MODALITY KEYWORDS
    MODALITIES = ['MRI', 'CT', 'T1', 'T2', 'X-ray', 'Ultrasound', 'Flair']

    # MODALITY TEMPLATES
    modality_question_open_templates = ["What type of medical image is this?", "What imaging modality was used?", "What is the modality by which the image was taken?", "What kind of scan is this?", "How was this image taken", "What type of imaging modality is seen in this image?", "What is the modality used?", "What imaging method was used?", "What modality is this?"]
    modality_question_closed_templates = ["Is this a {}?", "Is the image an {}?"]

    # PLANE KEYWORDS
    PLANES = ['axial', 'coronal', 'supratentorial', 'posteroanterior']
    #PLANES = ['axial', 'coronal']


    # PLANE TEMPLATES
    plane_question_open_templates = ["What is the scanning plane of this image?", "In what plane is this image scanned?", "In what plane is this image oriented?", "Which plane is this image taken?", "What is the name of this image's plane?", "How is the image oriented?", "What image plane is this?", "What plane are we in?"]
    plane_question_closed_templates = ["Is this a {} plane?", "Is this a {} image?", "Is this a {} section?", "Was this image taken in {} format?"]

    # PRESENCE KEYWORDS
    PRESENCE = ['pneumothorax', 'fracture', 'hernia', 'edema', 'hematoma', 'cyst', 'hemorrhage', 'lymphadenopathy', 'pneumoperitoneum']

    # PRESENCE TEMPLATES
    presence_question_closed_templates = ["Is there evidence of a {}?", "Is there a {}", "Is a {} present?"]

    # SHAPE REQUIRED WORDS
    SHAPE_REQUIRED = ['kidney', 'larynx', 'treachea', 'spine', 'spleen']
    SHAPE_KEYWORDS = ['irregular', 'oval', 'circular']
    SHAPE_TEMPLATES = ['What is the shape of the {} in this picture?']

    

    caption_path = os.path.join(PATH_TO_ROCO_DATA, "roco-dataset", "data", "train", "radiology", "captions.txt")
    semtype_path = os.path.join(PATH_TO_ROCO_DATA, "roco-dataset", "data", "train", "radiology", "semtypes.txt")
    keywords_path = os.path.join(PATH_TO_ROCO_DATA, "roco-dataset", "data", "train", "radiology", "keywords.txt")
    images_path = os.path.join(PATH_TO_ROCO_DATA, "roco-dataset", "data", "train", "radiology", "images")

    # Create category buckets
    ORGAN_SYSTEM_OPEN = QuestionCategoryBucket("Organ", ORGAN_SYSTEMS, organ_system_question_open_templates, "open", seed)
    ORGAN_SYSTEM_CLOSED = QuestionCategoryBucket("Organ", ORGAN_SYSTEMS, organ_system_question_closed_templates, "closed", seed)
    ORGAN_OPEN = QuestionCategoryBucket("Organ", ORGANS, organ_question_open_templates, "open", seed)
    ORGAN_CLOSED = QuestionCategoryBucket("Organ", ORGANS, organ_question_closed_templates, "closed", seed)

    MODALITY_OPEN = QuestionCategoryBucket("Modality", MODALITIES, modality_question_open_templates, "open", seed)
    MODALITY_CLOSED = QuestionCategoryBucket("Modality", MODALITIES, modality_question_closed_templates, "closed", seed)
    SHAPE_OPEN = SpecificQuestionCategoryBucket(SHAPE_REQUIRED, "Shape", SHAPE_KEYWORDS, SHAPE_TEMPLATES, "open", seed)
    PLANE_OPEN = QuestionCategoryBucket("Plane", PLANES, plane_question_open_templates, "open", seed)
    PLANE_CLOSED = QuestionCategoryBucket("Plane", PLANES, plane_question_closed_templates, "closed", seed)
    PRESENCE_CLOSED = QuestionCategoryBucket("Presence", PRESENCE, presence_question_closed_templates, "closed", seed)


    question_buckets = [ORGAN_SYSTEM_OPEN, ORGAN_SYSTEM_OPEN, ORGAN_OPEN, ORGAN_CLOSED, MODALITY_OPEN, MODALITY_CLOSED, PLANE_OPEN, PLANE_CLOSED]
    
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

    col_names = ['q_type', 'image_id', 'question', 'answer', 'question_type']
    row_data = []

    answer_counts = {} # key = answer, value = # occurences
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
                    num_answer_occurrences = answer_counts.get(answer, 0)
                    row_data.append([q_bucket.q_category, roco_id + ".jpg", question, answer, q_bucket.q_type])
                    answer_counts[answer] = answer_counts.get(answer, 0) + 1

    def get_stratified_split(rows, split_fraction = 0.2, seed=88):
        import random
        indices = []
        random.seed(seed)
        category_to_index = {}
        for index, row in enumerate(rows):
            if row[0] not in category_to_index:
                category_to_index[row[0]] = []
            category_to_index[row[0]] += [index]     

        # Sample according to split fraction
        for category in category_to_index:
            indices.extend(random.sample(category_to_index[category], int(len(category_to_index[category]) * split_fraction)))
        return indices

    indices = get_stratified_split(row_data)
    train_row_data = []
    test_row_data = []
    for i in range(len(row_data)):
        if i in indices:
            train_row_data.append(row_data[i])
        else:
            test_row_data.append(row_data[i])

    # For now, just use all data for retrieval
    train_df = pd.DataFrame(row_data)
    test_df = pd.DataFrame(row_data)
    train_df.columns = col_names
    test_df.columns = col_names

    os.makedirs(os.path.join(sys.argv[2], "ROCO"), exist_ok=True)
    train_df.to_csv(os.path.join(sys.argv[2], 'train.csv'), index=False)
    test_df.to_csv(os.path.join(sys.argv[2], 'test.csv'), index=False)