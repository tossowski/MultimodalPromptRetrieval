from transformers import T5Tokenizer, T5ForConditionalGeneration
from SLAKE import VQASLAKEFeatureDataset
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train a model", action="store_true")
parser.add_argument("--test", help="test a model", action="store_true")
args = parser.parse_args()


tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

max_source_length = 512
max_target_length = 128

dataset_train = VQASLAKEFeatureDataset("train", "data/SLAKE")
dataset_test = VQASLAKEFeatureDataset("test", "data/SLAKE")
train_loader = DataLoader(dataset_train, 16, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset_test, 16, shuffle=True, num_workers=2)


learning_rate = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if args.train:
    for batch_num, batch in enumerate(train_loader):
        print(batch_num)
        task_prefixes = [f"Answer the {x} question: " for x in batch['task']]
        encoding = tokenizer(
        [task_prefixes[i] + batch['question'][i] for i in range(len(batch['question']))],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
        )

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = tokenizer(
        batch['answer'], padding="longest", max_length=max_target_length, truncation=True
        )

        labels = target_encoding.input_ids
        labels = torch.tensor(labels)
        labels[labels == tokenizer.pad_token_id] = -100

        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, "models/model.pt")

# Test
if args.test:
    checkpoint = torch.load("models/model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    correct = defaultdict(int)
    performance = defaultdict(int)
    total = defaultdict(int)
    for batch in test_loader:
        task_prefixes = [f"Answer the {x} question: " for x in batch['task']]
        encoding = tokenizer(
        [task_prefixes[i] + batch['question'][i] for i in range(len(batch['question']))],
        padding="longest",
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
        )

        output_sequences = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        do_sample=False,  # disable sampling to test if batching affects output
        )

        predicted_answers = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        
        for i in range(len(predicted_answers)):
            if predicted_answers[i] == batch["answer"][i]:
                correct[batch["task"][i]] += 1
            total[batch["task"][i]] += 1
    for key in correct:
        performance[key] = correct[key] / total[key]
    print(performance)
    print(sum(correct.values())/sum(total.values()))
        #print([batch['answer'][i] + "|||" + predicted_answers[i] for i in range(len(predicted_answers))])

