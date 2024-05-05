import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch


class FoodsDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        input_text, target_text = item.input_text, item.target_text

        if not isinstance(input_text, str) or pd.isna(input_text):
            print(f"Warning: Non-string or NaN input_text at index {idx}. Using placeholder.")
            input_text = ""

        if not isinstance(target_text, str) or pd.isna(target_text):
            print(f"Warning: Non-string or NaN target_text at index {idx}. Using placeholder.")
            target_text = ""

        encoding = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True,
                                  return_tensors="pt")
        target_encoding = self.tokenizer(target_text, max_length=self.max_length, padding='max_length', truncation=True,
                                         return_tensors="pt")

        inputs = encoding.input_ids.squeeze()
        targets = target_encoding.input_ids.squeeze()

        return {"input_ids": inputs, "attention_mask": encoding.attention_mask.squeeze(), "labels": targets}


# Load your dataset
data_path = 'combined_foods.csv'  # Make sure to update this path
data = pd.read_csv(data_path)
data["input_text"] = "predict ingredients and type: " + data["Category"]
data["target_text"] = data["Components"] + " Type: " + data["Type"]
train_data, val_data = train_test_split(data[["input_text", "target_text"]], test_size=0.1)

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Prepare the dataset for training and validation
train_dataset = FoodsDataset(tokenizer, train_data)
val_dataset = FoodsDataset(tokenizer, val_data)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Start training
trainer.train()

#Save the trained model
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
