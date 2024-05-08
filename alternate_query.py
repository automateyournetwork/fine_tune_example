import torch
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline

# Load tokenizer directly from "aicvd"
tokenizer = AutoTokenizer.from_pretrained("aicvd")

# Load fine-tuned model directly from "aicvd"
model = LlamaForCausalLM.from_pretrained("aicvd", torch_dtype=torch.float16, device_map="auto")
model.to("cuda")

# Set up the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,  # Use CUDA device
    max_length=512,
    do_sample=True,
    top_p=0.95,
    temperature=0.8,
    repetition_penalty=1.2
)

# Test questions
questions = [
    "What is the out-of-band VLAN in FlexPod?",
    "What is the subnet for VLAN 1020?",
    "What is the name of VLAN 1020?",
    "If I was configuring a system on VLAN 1020 for out-of-band, what is my default gateway?"
]

# File to save output
output_file = "alternate_model_output.txt"

# Generate responses and write to file
role = "You are an expert on the Cisco Validated Design FlexPod Datacenter with Generative AI Inferencing Design and Deployment Guide."
with open(output_file, "w") as f:
    for question in questions:
        prompt = f"{role} {question}"
        response = pipe(prompt, num_return_sequences=1)[0]['generated_text']
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {response}\n\n")
        print(f"Question: {question}")
        print(f"Answer: {response}\n")
