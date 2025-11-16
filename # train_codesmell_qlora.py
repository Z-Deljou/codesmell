# train_codesmell_qlora.py
# -*- coding: utf-8 -*-

import os
import gc
import zipfile
import shutil
import json

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# -----------------------------
# تنظیمات کلی
# -----------------------------
BATCH_SIZE = 8
MAX_LEN = 3000
NUM_EPOCHS = 1
MODEL_NAME = "HuggingFaceTB/SmolLM-1.7B"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB_DISABLED"] = "true"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flush():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# -----------------------------
# آماده‌سازی دیتاست
# -----------------------------
def prepare_dataset():
    """
    اگر فولدر prepared_dataset وجود نداشته باشد، دیتاست را دانلود و unzip می‌کند.
    """
    if not os.path.exists("prepared_dataset"):
        print("Downloading dataset...")
        if os.path.exists("prepared_dataset"):
            shutil.rmtree("prepared_dataset")
        if os.path.exists("prepared_dataset.zip"):
            os.remove("prepared_dataset.zip")

        import gdown

        file_id = "1OjBBLcPOK4XysDuhU57TrBAKMlzJGrEp"
        gdown.download(
            id=file_id,
            output="prepared_dataset.zip",
            quiet=False,
        )

        print("Unzipping dataset...")
        with zipfile.ZipFile("prepared_dataset.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("Dataset prepared.")
    else:
        print("Dataset already exists.")


# -----------------------------
# دیتاست PyTorch
# -----------------------------
class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        code_path = os.path.join("prepared_dataset", "output_code", row["file_path"])

        with open(code_path, "r", encoding="utf-8") as f:
            code = f.read()

        labels = torch.tensor(row["encoded_labels"], dtype=torch.float)

        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            add_special_tokens=True,
        )

        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs["labels"] = labels

        return inputs


# -----------------------------
# متریک‌ها
# -----------------------------
def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions))
    preds = (preds > 0.5).int()
    labels = torch.tensor(p.label_ids)

    accuracy = (preds == labels).float().mean().item()

    true_positive = (preds * labels).sum(dim=0).float()
    predicted_positive = preds.sum(dim=0).float()
    actual_positive = labels.sum(dim=0).float()

    epsilon = 1e-7
    precision = (true_positive / (predicted_positive + epsilon)).mean().item()
    recall = (true_positive / (actual_positive + epsilon)).mean().item()
    f1_score = (
        2 * precision * recall / (precision + recall + epsilon)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def main():
    
    prepare_dataset()

    df = pd.read_csv(
        "prepared_dataset/dataset.csv", header=None, names=["file_path", "codesmells"]
    )
    df.rename(columns={"codesmells": "labels"}, inplace=True)
    df["labels"] = df["labels"].apply(lambda x: x.split(","))

    all_labels = set(label for sublist in df["labels"] for label in sublist)
 
    all_labels = sorted(list(all_labels))

    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    id2label = {i: label for i, label in enumerate(all_labels)}

    def encode_labels(labels):
        encoded = [0] * len(label_to_idx)
        for label in labels:
            encoded[label_to_idx[label]] = 1
        return encoded

    df["encoded_labels"] = df["labels"].apply(encode_labels)

    # حذف اولین ردیف مثل نوت‌بوک اصلی
    df = df.iloc[1:].copy()

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=0.002, random_state=42, shuffle=True
    )

    # ۲) توکنایزر و مدل
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(all_labels),
        quantization_config=quantization_config,
        problem_type="multi_label_classification",
        low_cpu_mem_usage=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # ۳) آماده‌سازی QLoRA
    model.train()
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # ۴) دیتاست‌ها و کالیتور
    train_dataset = CodeDataset(train_df, tokenizer)
    test_dataset = CodeDataset(test_df, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ۵) آرگومان‌های آموزش (با ذخیره‌ی چک‌پوینت)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=50,
        learning_rate=2e-5,
        save_strategy="steps",
        save_steps=20,          # هر ۲۰ استپ چک‌پوینت
        save_total_limit=2,     # فقط دو چک‌پوینت آخر
        warmup_steps=200,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        max_steps=800,
        weight_decay=0.01,
        gradient_accumulation_steps=4,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    model.config.use_cache = False

    
    trainer.train(resume_from_checkpoint=True)

    model.config.use_cache = True

    results = trainer.evaluate()
    print("Evaluation results:", results)


    output_model_dir = "./trained_code_smell_smolLM"
    os.makedirs(output_model_dir, exist_ok=True)
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    
    labels_info = {
        "all_labels": all_labels,
        "label_to_idx": label_to_idx,
        "id2label": id2label,
    }
    with open(os.path.join(output_model_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels_info, f, ensure_ascii=False, indent=2)

    print(f"Model, tokenizer and labels saved to {output_model_dir}")


if __name__ == "__main__":
    main()
