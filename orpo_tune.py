import gc
import os
import json
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format
from huggingface_hub import HfApi, create_repo, upload_file

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# Model
base_model = "meta-llama/Meta-Llama-3-8B"
new_model = "OrpoLlama-3-8B"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)
model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)

# Load dataset from JSONL file
def load_jsonl_dataset(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

jsonl_file_path = "training_dataset.jsonl"  # Replace with your actual JSONL file path
dataset = load_jsonl_dataset(jsonl_file_path)

# Save the modified dataset back to JSONL
def save_jsonl_dataset(dataset, file_path):
    with open(file_path, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

training_dataset_path = "training_dataset.jsonl"
save_jsonl_dataset(dataset, training_dataset_path)

# Create and configure the dataset for training
print("dataset load")
dataset = load_dataset('json', data_files={'train': training_dataset_path}, split='all')
print("dataset shuffle")
dataset = dataset.shuffle(seed=42)

def format_chat_template(row):
    role = "You are an expert on the Cisco Validated Design FlexPod Datacenter with Generative AI Inferencing Design and Deployment Guide."
    row["chosen"] = f'{role} {row["chosen"]}'
    row["rejected"] = f'{role} {row["rejected"]}'
    row["role"] = role
    return row

print("dataset map")
dataset = dataset.map(
    format_chat_template,
    num_proc=os.cpu_count() // 2,
    batched=False
)

print("dataset train_test_split")
dataset = dataset.train_test_split(test_size=0.01)

orpo_args = ORPOConfig(
    learning_rate=5e-5,  # Lower learning rate
    beta=0.1,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,  # Smaller batch size
    per_device_eval_batch_size=2,   # Smaller batch size for evaluation
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=3,  # Increase the number of epochs
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    warmup_steps=10,
    report_to="wandb",
    output_dir="./results/",
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)
print("training model")
trainer.train()
print("saving model")
trainer.save_model(new_model)

# Flush memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()

# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model, tokenizer = setup_chat_format(model, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(model, new_model)
print("merge and unload model")
model = model.merge_and_unload().to("cuda")

# Define the path to your files
adapter_config_path = "aicvd/adapter_config.json"
adapter_model_path = "aicvd/adapter_model.safetensors"
repo_name = "automateyournetwork/aicvd"
# Push model and tokenizer to the Hugging Face Hub
model.push_to_hub(repo_name, use_temp_dir=False)
tokenizer.push_to_hub(repo_name, use_temp_dir=False)

# Initialize the HfApi client
api = HfApi()

# Ensure the repository exists
create_repo(repo_name, exist_ok=True)

# Upload files individually
api.upload_file(
    path_or_fileobj=adapter_config_path,
    path_in_repo="adapter_config.json",
    repo_id=repo_name,
    repo_type="model"
)

api.upload_file(
    path_or_fileobj=adapter_model_path,
    path_in_repo="adapter_model.safetensors",
    repo_id=repo_name,
    repo_type="model"
)

api.upload_file(
    path_or_fileobj="training_args.bin",
    path_in_repo="training_args.bin",
    repo_id=repo_name,
    repo_type="model"
)

def ask_model(question, model, tokenizer, device, max_length=128, num_beams=3):
    role = "You are an expert on the Cisco Validated Design FlexPod Datacenter with Generative AI Inferencing Design and Deployment Guide."
    context = f"{question}"
    inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with autocast():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,  # Added to reduce repetitive phrases
                num_return_sequences=1   # Generating only one response
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    main()
