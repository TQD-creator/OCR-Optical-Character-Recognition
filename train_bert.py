import os
import json
import glob
import re
import torch
import numpy as np
import evaluate
from torch.utils.data import Dataset, random_split, Subset
from PIL import Image
from difflib import SequenceMatcher
from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer

os.environ['TORCH_DISABLE_DYNAMO'] = '1'

print("[PHASE 1] Loading core functions and the real SROIE dataset for BERT...")

def get_bio_label(text, entities):
    text_upper = text.upper().strip()
    text_clean = re.sub(r'[^A-Z0-9]', '', text_upper)
    if len(text_clean) < 2: return 'O'
    for key, value in entities.items():
        if not value: continue
        val_upper = str(value).upper().strip()
        val_clean = re.sub(r'[^A-Z0-9]', '', val_upper)
        if (text_upper in val_upper or val_upper in text_upper) and len(text_upper) > 2: return key.upper()
        if (text_clean in val_clean or val_clean in text_clean) and len(text_clean) > 4: return key.upper()
        ratio = SequenceMatcher(None, text_clean, val_clean).ratio()
        if ratio > 0.85 and len(text_clean) > 5: return key.upper()
    return 'O'

class BertSROIEDataset(Dataset):
    """Dataset for BERT (without bounding box information)"""
    def __init__(self, task1_dir, task2_dir, image_dir, tokenizer, max_seq_length=512):
        all_task1_files = glob.glob(os.path.join(task1_dir, "*.txt"))
        self.task1_files = []
        self.task2_dir = task2_dir
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_map = {"O": 0, "COMPANY": 1, "DATE": 2, "ADDRESS": 3, "TOTAL": 4}

        for f in all_task1_files:
            base_name = os.path.basename(f)
            json_path = os.path.join(self.task2_dir, base_name)
            img_path = os.path.join(self.image_dir, base_name.replace('.txt', '.jpg'))
            if os.path.exists(json_path) and os.path.exists(img_path):
                self.task1_files.append(f)
        self.task1_files.sort()

    def __len__(self): return len(self.task1_files)

    def __getitem__(self, item):
        file_path = self.task1_files[item]
        base_name = os.path.basename(file_path).replace('.txt', '')
        json_path = os.path.join(self.task2_dir, base_name + ".txt")
        img_path = os.path.join(self.image_dir, base_name + ".jpg")

        with Image.open(img_path) as img: width, height = img.size
        with open(json_path, 'r', encoding='utf-8') as f: entities = json.load(f)

        all_input_ids, all_labels = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                parts = line.strip().split(',', maxsplit=8)
                if len(parts) < 9: continue
                text = parts[8]
                label_id = self.label_map.get(get_bio_label(text, entities), 0)

                for i, token in enumerate(self.tokenizer.tokenize(text)):
                    all_input_ids.append(self.tokenizer.convert_tokens_to_ids(token))
                    all_labels.append(label_id if i == 0 else -100)

        all_input_ids = all_input_ids[:self.max_seq_length - 2]
        all_labels = all_labels[:self.max_seq_length - 2]

        input_ids = [self.tokenizer.cls_token_id] + all_input_ids + [self.tokenizer.sep_token_id]
        labels = [-100] + all_labels + [-100]

        pad_len = self.max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask = [1] * (self.max_seq_length - pad_len) + [0] * pad_len

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask), 
            'labels': torch.tensor(labels)
        }

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Local paths assuming your structure is ./data/raw/...
base_path = './data/raw'
full_dataset = BertSROIEDataset(
    task1_dir=f'{base_path}/task1_train/', task2_dir=f'{base_path}/task2_train/',
    image_dir=f'{base_path}/task1_train/', tokenizer=tokenizer
)

# Creating the 3-way split we discussed
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
print(f"[SUCCESS] Loaded {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples.")

print("[PHASE 2] Setting up tools and injecting hardcoded parameters...")

metric = evaluate.load("seqeval")
label_list = ["O", "COMPANY", "DATE", "ADDRESS", "TOTAL"]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[label_list[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

def model_init():
    return BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(label_list))

# --- YOUR HARDCODED PARAMETERS ---
fixed_lr = 1.5591908316711295e-05
fixed_batch_size = 2
fixed_epochs = 3
fixed_weight_decay = 0.07877919257368415

print("\n[PHASE 3] Bypassing Optuna. Training BERT model directly with fixed parameters...")

bert_args = TrainingArguments(
    output_dir="./bert_final_model",
    learning_rate=fixed_lr,
    per_device_train_batch_size=fixed_batch_size,
    num_train_epochs=fixed_epochs,
    weight_decay=fixed_weight_decay,
    fp16=False,
    dataloader_pin_memory=False,
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

# Combine train and val for final training to maximize data usage before testing
combined_dataset = Subset(full_dataset, list(range(len(train_dataset))) + list(range(len(train_dataset), len(train_dataset) + len(val_dataset))))

bert_trainer = Trainer(
    model_init=model_init, 
    args=bert_args, 
    train_dataset=combined_dataset,
    eval_dataset=test_dataset, 
    processing_class=tokenizer, 
    compute_metrics=compute_metrics
)

# Train the model exactly once using your exact settings
bert_trainer.train()

print("\n[PHASE 4] Evaluating on TEST set (unseen data)...")
test_metrics = bert_trainer.evaluate(eval_dataset=test_dataset)
print(f"\n[TEST RESULTS]")
print(f"  Precision: {test_metrics['eval_precision']:.4f}")
print(f"  Recall: {test_metrics['eval_recall']:.4f}")
print(f"  F1 Score: {test_metrics['eval_f1']:.4f}")
print(f"  Accuracy: {test_metrics['eval_accuracy']:.4f}")

print("\n[PHASE 5] Saving the BERT model to disk...")
save_directory = "./saved_bert_model"
bert_trainer.save_model(save_directory) 
tokenizer.save_pretrained(save_directory)

# Save test metrics
metrics_file = os.path.join(save_directory, "test_metrics.json")
with open(metrics_file, 'w') as f:
    json.dump(test_metrics, f, indent=4)

print(f"\n[SUCCESS] BERT training complete. Your model is ready to use at: {save_directory}")