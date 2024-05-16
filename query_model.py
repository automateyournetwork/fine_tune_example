import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM
from torch.cuda.amp import autocast
from langchain_community.llms import Ollama

def main():
    model_dir = "./aicvd"
    base_model_name = "meta-llama/Meta-Llama-3-8B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = Ollama(model="llama3")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load the fine-tuned model
    try:
        torch.cuda.empty_cache()  # Clear cached memory
        base_model_instance = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16).to(device)

        # Resize token embeddings if necessary
        if len(tokenizer) != base_model_instance.config.vocab_size:
            print("Resizing token embeddings to match tokenizer's vocabulary size.")
            base_model_instance.resize_token_embeddings(len(tokenizer))

        fine_tuned_model = PeftModelForCausalLM.from_pretrained(base_model_instance, model_dir)
        fine_tuned_model = fine_tuned_model.to(device)
        print("Fine-tuned model loaded successfully.")
    except Exception as e:
        print(f"Failed to load fine-tuned model: {e}")
        return

    # Define the questions based on the templates
    question_templates = [
        "What is the VLAN ID for the <VLAN Name>?",
        "Which subnet is associated with VLAN ID <VLAN ID>?",
        "Identify the gateway for the <VLAN Name>.",
        "Describe the purpose of the <VLAN Name>.",
        "Explain the purpose of VLAN ID <VLAN ID>.",
        "What subnet does <VLAN Name> use?",
        "What gateway is assigned to VLAN ID <VLAN ID>?",
        "How is the <VLAN Name> utilized?",
        "What function does the <VLAN Name> serve?",
        "What role does VLAN ID <VLAN ID> play?",
        "Which IP subnet is configured for <VLAN Name>?",
        "Specify the gateway IP for <VLAN Name>.",
        "For what purpose is the <VLAN Name> used?",
        "What is the VLAN ID assigned to <VLAN Name>?",
        "What is the subnet mask for VLAN ID <VLAN ID>?",
        "What is the default gateway for the <VLAN Name>?",
        "What is the primary function of the <VLAN Name>?",
        "What is the primary purpose of VLAN ID <VLAN ID>?",
        "Which subnet is allocated to <VLAN Name>?",
        "Identify the default gateway for VLAN ID <VLAN ID>.",
        "What is the assigned VLAN ID for the <VLAN Name>?",
        "What IP subnet does VLAN ID <VLAN ID> correspond to?",
        "What is the configured gateway for the <VLAN Name>?",
        "What is the main purpose of the <VLAN Name>?",
        "Which subnet is utilized by VLAN ID <VLAN ID>?",
        "What VLAN ID is used for <VLAN Name> in FlexPod?",
        "What subnet is used by <VLAN Name> for FlexPod?",
        "What is the gateway IP for VLAN ID <VLAN ID>?",
        "What does the <VLAN Name> VLAN do?",
        "What is the function of VLAN ID <VLAN ID>?",
        "What subnet is assigned to <VLAN Name>?",
        "What is the default gateway for VLAN ID <VLAN ID>?",
        "How is VLAN ID <VLAN ID> utilized?",
        "What is the IP subnet for <VLAN Name>?",
        "What gateway does <VLAN Name> use?",
        "What is the purpose of VLAN ID <VLAN ID> in FlexPod?",
        "Which IP subnet is used for VLAN ID <VLAN ID>?",
        "What is the purpose of the VLAN ID <VLAN ID>?",
        "What is the role of <VLAN Name> in FlexPod?",
        "Identify the subnet for VLAN ID <VLAN ID>.",
        "What is the purpose of <VLAN Name> VLAN?",
        "What is the gateway for <VLAN Name> in FlexPod?",
        "What is the VLAN ID of <VLAN Name> in FlexPod?",
        "What is the IP subnet of VLAN ID <VLAN ID>?",
        "Explain the role of <VLAN Name> in FlexPod.",
        "What subnet does VLAN ID <VLAN ID> use?",
        "What is the primary role of <VLAN Name>?",
        "What function does VLAN ID <VLAN ID> serve in FlexPod?",
        "What is the assigned IP subnet for <VLAN Name>?",
        "Which gateway is used for VLAN ID <VLAN ID>?"
    ]

    # Define the actual dataset
    dataset = [
        {"VLAN ID": "2", "Name": "Native-VLAN", "Usage": "Use VLAN 2 as native VLAN instead of default VLAN (1)", "IP Subnet": "N/A", "IP Gateway": "N/A"},
        {"VLAN ID": "1020", "Name": "OOB-MGMT-VLAN", "Usage": "Out-of-band management VLAN to connect management ports for various devices", "IP Subnet": "10.102.0.0/24", "IP Gateway": "10.102.0.254"},
        {"VLAN ID": "1021", "Name": "IB-MGMT-VLAN", "Usage": "In-band management VLAN utilized for all in-band management connectivity for example ESXi hosts or VM management", "IP Subnet": "10.102.1.0/24", "IP Gateway": "10.102.1.254"},
        {"VLAN ID": "1022", "Name": "OCP-MGMT", "Usage": "OCP management traffic VLAN used in place of VM-Traffic VLAN", "IP Subnet": "10.102.2.0/24", "IP Gateway": "10.102.2.254"},
        {"VLAN ID": "3050", "Name": "NFS-VLAN", "Usage": "NFS VLAN for mounting datastores in ESXi servers for VMs", "IP Subnet": "192.168.50.0/24", "IP Gateway": "N/A"},
        {"VLAN ID": "3010", "Name": "iSCSI-A", "Usage": "iSCSI-A path for storage traffic including boot-from-san traffic", "IP Subnet": "192.168.10.0/24", "IP Gateway": "N/A"},
        {"VLAN ID": "3020", "Name": "iSCSI-B", "Usage": "iSCSI-B path for storage traffic including boot-from-san traffic", "IP Subnet": "192.168.20.0/24", "IP Gateway": "N/A"},
        {"VLAN ID": "3030", "Name": "NVMe-TCP-A", "Usage": "NVMe-TCP-A path when using NVMe-TCP", "IP Subnet": "192.168.30.0/24", "IP Gateway": "N/A"},
        {"VLAN ID": "3040", "Name": "NVMe-TCP-B", "Usage": "NVMe-TCP-B path when using NVMe-TCP", "IP Subnet": "192.168.40.0/24", "IP Gateway": "N/A"},
        {"VLAN ID": "3000", "Name": "vMotion", "Usage": "VMware vMotion traffic", "IP Subnet": "192.168.0.0/24", "IP Gateway": "N/A"},
        {"VLAN ID": "3052", "Name": "OCP-NFS", "Usage": "NFS VLAN for OCP persistent storage and OCP cluster and support VMs", "IP Subnet": "192.168.52.0/24", "IP Gateway": "N/A"},
        {"VLAN ID": "3032", "Name": "OCP-NVMe-TCP-A", "Usage": "NVMe-TCP-A path when using NVMe-TCP for persistent storage", "IP Subnet": "192.168.32.0/24", "IP Gateway": "N/A"},
        {"VLAN ID": "3042", "Name": "OCP-NVMe-TCP-B", "Usage": "NVMe-TCP-B path when using NVMe-TCP for persistent storage", "IP Subnet": "192.168.42.0/24", "IP Gateway": "N/A"}
    ]

    # Generate questions based on the dataset and templates
    questions = generate_questions_from_dataset(dataset, question_templates)

    output_file = "model_output.txt"
    # Clear the file first
    with open(output_file, "w") as file:
        file.write("")

    test_model("Fine-Tuned Model", fine_tuned_model, tokenizer, questions, device, output_file)
    test_model_with_llm("Base Model", llm, questions, output_file)

def generate_questions_from_dataset(dataset, question_templates):
    questions = []
    for entry in dataset:
        for template in question_templates:
            question = template.replace("<VLAN Name>", entry["Name"]).replace("<VLAN ID>", entry["VLAN ID"])
            questions.append(question)
    return questions

def test_model(model_name, model, tokenizer, questions, device, output_file):
    with open(output_file, "a") as file:  # Open file in append mode
        file.write(f"\nTesting {model_name}:\n")
        print(f"\nTesting {model_name}:")
        for question in questions:
            answer = ask_model(question, model, tokenizer, device)
            output = f"{question}\n\n{answer}\n"
            file.write(output)
            print(output)

def test_model_with_llm(model_name, llm, questions, output_file):
    with open(output_file, "a") as file:  # Open file in append mode
        file.write(f"\nTesting {model_name}:\n")
        print(f"\nTesting {model_name}:")
        for question in questions:
            response = llm.invoke(question)
            output = f"{question}\n\n{response}\n"
            file.write(output)
            print(output)

def ask_model(question, model, tokenizer, device, max_length=128, num_beams=3):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
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
