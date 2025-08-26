import torch
import os
import gc
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

!nvidia-smi

!pip install transformers datasets peft torch accelerate pdfplumber evaluate

import pdfplumber
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import numpy as np

# extract pdf text - need to handle errors cuz file might not exist
pdf_path = "stanford_notes.pdf"

try:
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Processing {len(pdf.pages)} pages...")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:  # some pages might be empty
                all_text += text + "\n\n"
            if i % 10 == 0:
                print(f"Processed page {i+1}/{len(pdf.pages)}")
    
    print(f"Extracted {len(all_text)} characters total")
    
    with open("stanford_notes.txt", "w", encoding="utf-8") as f:
        f.write(all_text)
    
    print("PDF text extraction complete. Sample:")
    print(all_text[:500])
    
except FileNotFoundError:
    print(f"Error: {pdf_path} not found. Please upload your PDF file.")
    # fallback sample text if no pdf
    all_text = """This is sample text for testing the fine-tuning pipeline. 
    In machine learning, we often need to preprocess data before training models.
    Fine-tuning involves adapting a pre-trained model to a specific task or domain.
    """ * 100  
    print("Using sample text for demonstration.")

# clean up the messy pdf text
print("Cleaning text...")
clean_text = all_text.replace("\n", " ")
clean_text = " ".join(clean_text.split())  # get rid of extra spaces

# split by sentences - better for training than random chunks
import re
sentences = re.split(r'[.!?]+', clean_text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # filter out tiny sentences

print(f"Split into {len(sentences)} sentences")

with open("stanford_notes_clean.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sentences))

# make dataset from our sentences
print("Creating dataset...")
dataset_dict = {"text": sentences}
dataset = Dataset.from_dict(dataset_dict)

# train/test split
dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")
print("Sample text:", dataset["train"][0]["text"][:100])

# load tokenizer
model_name = "gpt2"
print(f"Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# gpt2 doesn't have pad token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# tokenize function - had some issues with tensor shapes before
def tokenize_function(examples):
    result = tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length",
        max_length=256,
        return_tensors=None  # return lists not tensors
    )
    # for causal lm labels = input_ids
    result["labels"] = result["input_ids"].copy()
    return result

# apply tokenization to dataset
print("Tokenizing dataset...")
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"],
    desc="Tokenizing"
)

print(f"Tokenized dataset: {tokenized_datasets}")
print(f"Sample tokenized length: {len(tokenized_datasets['train'][0]['input_ids'])}")

# load base gpt2 model
print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

# might need this if we added tokens
model.resize_token_embeddings(len(tokenizer))

# lora config - efficient fine-tuning
print("Setting up LoRA...")
lora_config = LoraConfig(
    r=8,  # rank
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # which layers to adapt
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# prep model for lora
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# check how many params we're actually training
model.print_trainable_parameters()

# data collator for batching
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # causal lm not masked lm
)

# training config
training_args = TrainingArguments(
    output_dir="./stanford_notes_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4 if torch.cuda.is_available() else 2,
    per_device_eval_batch_size=4 if torch.cuda.is_available() else 2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,  # higher lr works better for lora
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    report_to="none",
    seed=42,
    data_seed=42,
)

# setup trainer
print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# actually train the thing
print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
    
    # save model
    trainer.save_model("./stanford_notes_final")
    tokenizer.save_pretrained("./stanford_notes_final")
    print("Model saved!")
    
except Exception as e:
    print(f"Training error: {e}")
    print("You may need to reduce batch size or sequence length if you encounter OOM errors")

# test it out
print("\nTesting the fine-tuned model...")
try:
    # eval mode
    model.eval()
    
    # try generating text
    test_prompt = "In machine learning"
    inputs = tokenizer.encode(test_prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text:\n{generated_text}")
    
except Exception as e:
    print(f"Generation error: {e}")

# cleanup to free memory
print("\nCleaning up memory...")
del model
del trainer
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("Done!")

# function to load model later if needed
def load_finetuned_model(model_path="./stanford_notes_final"):
    """
    load the saved model for inference later
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        print("Fine-tuned model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# uncomment to test loading:
# loaded_model, loaded_tokenizer = load_finetuned_model()
