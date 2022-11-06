import json
from PIL import Image
import torch
import os
import pickle
import clip
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence



class VQADataset(Dataset):
    def __init__(self, name, dataroot, device = "cuda" if torch.cuda.is_available() else "cpu"):
        super(VQADataset, self).__init__()
        self.name = name
        self.dataroot = dataroot
        self.entries = self._load_dataset(dataroot, name)
        self.device = device
        

        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        images_path = os.path.join(dataroot, f'images_{name}.pkl')
        if os.path.exists(images_path):
            print(f"Loading existing images from {images_path}")
            with open(images_path, 'rb') as f:
                self.images = pickle.load(f)
            print(f"Loaded {len(self.images)} existing images")
        else:
            print(f"Creating images file: {images_path}")
            image_dict = {}
            for entry in self.entries:
                if entry['image_name'] in image_dict:
                    continue
                image_path = os.path.join(dataroot, "imgs", entry['image_name'])
                image = Image.open(image_path)
                image = self.preprocess(image)
                image_dict[entry['image_name']] = image
            with open(images_path, 'wb') as f:
                pickle.dump(image_dict, f)
            with open(images_path, 'rb') as f:
                self.images = pickle.load(f)
            print(f"Loaded {len(self.images)} existing images")

    def add_labels(self, ans2label):
        for i in range(len(self.entries)):
            answer = self.entries[i]["answer"]
            self.entries[i]["labels"] = ans2label[answer]

    def _load_dataset(self, dataroot, name):
        data_path = os.path.join(dataroot, name + '.json')
        
        samples_all = json.load(open(data_path))
        samples = [sample for sample in samples_all if sample['q_lang']=="en"]

        entries = []
        for entry in samples:

            sample = {'image_name' : entry['img_name'],
                'question_id': str(entry['qid']),
                'question': entry['question'].lower(),
                'answer' : entry['answer'].lower(),
                'task': entry['content_type'],
                'question_type': entry['answer_type'].lower()}

            if entry['answer'] == '':
                continue
            entries.append(sample)

        
        return entries

    def filter_max_answers(self, num, answer_set = None, config=None):
        if answer_set == None:
            possible_open_answers = set([entry["answer"] for entry in self.entries if entry["question_type"] == "open"])
            possible_closed_answers = set([entry["answer"] for entry in self.entries if entry["question_type"] == "closed"])
            for answer in set.intersection(possible_open_answers, possible_closed_answers):
                possible_open_answers.remove(answer)
            print(f"There are {len(possible_open_answers)} open and {len(possible_closed_answers)} closed answers")
            answer_set = sorted(possible_open_answers)[:num//2] + sorted(possible_closed_answers)[:num//2]
        self.entries = [x for x in self.entries if x["answer"] in answer_set]
        #print(f"Filtered {num} answers. There are now {len(self.entries)} examples in dataset")
        return answer_set


    def filter(self, qtype_list, limit_num_examples = float("inf")):
        counts = {}
        new_entries = []
        
        for entry in self.entries:
            if entry["task"] in qtype_list:
                if entry["task"] not in counts:
                    counts[entry["task"]] = 0
                if counts[entry["task"]] >= limit_num_examples:
                    continue
                counts[entry["task"]] += 1
                new_entries.append(entry)
        self.entries = new_entries

    def get_question_by_id(self, qid):
        for i in range(len(self.entries)):
            if self.entries[i]["question_id"] == qid:
                return self.__getitem__(i)

    def create_retrieval_dataset(self, data_loader, prefix):
        embedding_path = os.path.join("cache", prefix, self.__class__.__name__, "embedding.pt")
        answer_path = os.path.join("cache", prefix, self.__class__.__name__, "answers.pkl")
        if os.path.exists(embedding_path) and os.path.exists(answer_path):
            
            self.retrieval_embeddings = torch.load(embedding_path)
            print(f"Loaded cached qa lookup embeddings from {embedding_path} ...")
            with open(answer_path, 'rb') as f:
                self.retrieval_answers = pickle.load(f)
                print(f"Loaded cached qa lookup answers from {answer_path} ...")
        else:
            os.makedirs(os.path.join("cache", prefix, self.__class__.__name__), exist_ok=True)
            print(f"Creating qa pairs in {os.path.join('cache', prefix, self.__class__.__name__)} ...")
            all_embeddings = []
            all_answers = []
            for batch in tqdm(data_loader):
                image_encoding = self.clip_model.encode_image(batch["image"].to(self.device))
                text_encoding = self.clip_model.encode_text(clip.tokenize(batch["question"]).to(self.device))
                answers = batch["answer"]
                all_embeddings.append(torch.cat([image_encoding,text_encoding], axis=1).detach())
                all_answers.extend(answers)

            
            #answers = torch.cat(answers, axis = 0)
            self.retrieval_embeddings = torch.cat(all_embeddings, axis=0)
            self.retrieval_answers = all_answers
            torch.save(self.retrieval_embeddings, embedding_path)
            with open(answer_path, 'wb') as f:
                pickle.dump(all_answers, f)

    def retrieve_closest_qa_pairs(self, batch):
        buckets = ["very unlikely", "unlikely", "maybe", "likely", "very likely", "certainly"]
        image_encoding = self.clip_model.encode_image(batch["image"].to(self.device))
        text_encoding = self.clip_model.encode_text(clip.tokenize(batch["question"]).to(self.device))
        combined = torch.cat([image_encoding,text_encoding], axis=1)
        dist_matrix = torch.cdist(combined, self.retrieval_embeddings)
        top15_closest_indices = torch.argsort(dist_matrix, axis = 1)[:,1:16]
        answers = [[self.retrieval_answers[x] for x in top15_closest_indices[i,:]] for i in range(len(top15_closest_indices))]
        prompts = []
        for row in answers:
            answer_counts = {}
            for answer in row:
                if answer not in answer_counts:
                    answer_counts[answer] = 0
                answer_counts[answer] += 1
            pred_answer = max(answer_counts, key = answer_counts.get)
            certainty = max(answer_counts.values())/sum(answer_counts.values())
            prompt = buckets[int(certainty * (len(buckets) - 1))]
            prompts.append(f"I believe the answer is {prompt} {pred_answer}")
        return prompts



    def __len__(self):
        return len(self.entries)
    

    def __getitem__(self, index):
        entry = self.entries[index]
        item = {}
        item["path_to_image"] = os.path.join(self.dataroot, "imgs", entry['image_name'])
        item['image'] = self.images[entry['image_name']]
        item['question'] = entry['question']
        item['answer'] = entry['answer']
        item['task'] = entry['task']
        item['question_id'] = entry['question_id']
        item['question_type'] = entry['question_type']

        if 'labels' in entry:
            item['labels'] = entry['labels']
        return item