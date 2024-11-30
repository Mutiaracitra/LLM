# Import libraries
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd
from datasets import Dataset

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Ensure tokenizer handles padding
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
csv_path = 'C:\IYKRA\week 4\weekly\LLM\data.csv'
df = pd.read_csv(csv_path)

# Convert dataset to Hugging Face Dataset format
# Assume the dataset has 'input' and 'output' columns
dataset = Dataset.from_pandas(df)

# Preprocess data
def tokenize_data(example):
    return tokenizer(
        example['input'], 
        text_target=example['output'], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_data, batched=True)

# Split dataset into train and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# Define Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # GPT-2 doesn't use masked language modeling
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Output directory for the model
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir="./logs",  # Directory for logs
    logging_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_gpt2')
tokenizer.save_pretrained('./fine_tuned_gpt2')

print("Model fine-tuning completed and saved!")
