import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

 # Detect Device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# Step 1: Load Cyberbullying Dataset
dataset_path = "set ur dataset path "
df = pd.read_csv(dataset_path)

# Step 2: Preprocess Data
df = df[['tweet_text', 'cyberbullying_type']]
df.dropna(inplace=True)
df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'label'}, inplace=True)
df['label'] = df['label'].astype(int)

# Data Augmentation (WordNet Synonym Replacement)
from textattack.augmentation import WordNetAugmenter
augmenter = WordNetAugmenter()
df["augmented_text"] = df["text"].apply(lambda x: augmenter.augment(x)[0])
df = pd.concat([df[["text", "label"]], df[["augmented_text", "label"]].rename(columns={"augmented_text": "text"})])

# Step 3: Split Data into Train & Test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Step 4: Load BERT Tokenizer & Model
model_name = "berta-base"  # Using BERTa instead of BERT
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Step 5: Tokenize the Data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).map(tokenize_function, batched=True)
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels}).map(tokenize_function, batched=True)

# Step 6: Fine-Tune BERT model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,  # Increased epochs depending upon the dataset size
    per_device_train_batch_size=32,  # Reduced batch size 32 64 128
    per_device_eval_batch_size=32,
    logging_dir="./logs",
    logging_steps=10,
    fp16=torch.cuda.is_available(),  # Enable FP16 training
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_hf",
    learning_rate=3e-5,  # learning rate can be increased or decrease on the basis of the dataset 
    weight_decay=0.01,  # Regularization to prevent overfitting
    lr_scheduler_type="linear",  # Learning rate decay
    warmup_steps=500,  # Warmup stabilization
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Handle Class Imbalance (Weighted Sampling)
from torch.utils.data import WeightedRandomSampler
class_counts = np.bincount(train_labels)
weights = 1.0 / class_counts
sample_weights = [weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_sampler=sampler  # Balanced sampling
)

print(" Training Started...")
trainer.train()
print(" Training Completed!")

# Step 7: Save the Trained Model
model.save_pretrained("./cyberbullying_model")
tokenizer.save_pretrained("./cyberbullying_model")
print(" Model saved successfully!")
