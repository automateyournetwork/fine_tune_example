import gc
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

# Configuration for GPU and torch dtype
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16

# Model details
base_model = "meta-llama/Meta-Llama-3-8B"
new_model = "aicvd"

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration
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

# Load base model with quantization configuration
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

# Apply LoRA adapter using get_peft_model
model = get_peft_model(model, peft_config)

file_path = "training_dataset.jsonl"
print("dataset load")
dataset = load_dataset('json', data_files={'train': file_path}, split='all')
print("dataset shuffle")
dataset = dataset.shuffle(seed=42)

# Apply chat template with ORPO-specific formatting
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

# ORPO Configuration
orpo_args = ORPOConfig(
    learning_rate=1e-4,
    beta=0.1,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=3,
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

# Load LoRA adapter into model
model = get_peft_model(model, peft_config)

# Save the LoRA adapter configuration file (adapter_config.json)
model.save_pretrained(new_model)

# Optionally merge and unload the LoRA adapter
print("merge and unload model")
model = model.merge_and_unload().to("cuda")

model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
