from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ===== Custom Dataset Loader =====
class MyTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        print(f"✅ Loaded {len(text)} characters from {file_path}")

        tokenized = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=block_size,
            return_tensors="pt"
        )
        self.input_ids = tokenized["input_ids"]
        self.attn_masks = tokenized["attention_mask"]

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_masks[idx],
            "labels": self.input_ids[idx],
        }

# ===== Load tokenizer and model =====
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Fix for GPT2 padding
model = GPT2LMHeadModel.from_pretrained("gpt2")

# ===== Load dataset =====
dataset_path = r"C:\Users\ngvan\gpt2-finetune\math_data.txt"
train_dataset = MyTextDataset(dataset_path, tokenizer)

# ===== Training arguments =====
training_args = TrainingArguments(
    output_dir="./gpt2-math-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=1,
    logging_steps=10
)

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# ===== Train! =====
trainer.train()
trainer.save_model("./gpt2-math-finetuned")
tokenizer.save_pretrained("./gpt2-math-finetuned")
