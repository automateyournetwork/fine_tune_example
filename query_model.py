import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftModelForCausalLM
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

    # Example inference to check the model
    questions = [
        "What is the VLAN ID for the out-of-band management VLAN in the FlexPod environment?",
        "What is the subnet for the out-of-band management VLAN in the FlexPod environment?",
        "What is the gateway for the out-of-band management VLAN in the FlexPod environment?",
        "What is the name for the out-of-band management VLAN in the FlexPod environment?",
        "What is the name of the VLAN ID 1020 in the FlexPod environment?",
        "What is the gateway address for the OOB-MGMT-VLAN",
        "What is the IP subnet used for the OOB-MGMT-VLAN in the FlexPod environment??",
        "What can you tell me about out of band management on the FlexPod?",
        "I am setting up out of band management on the FlexPod what is the VLAN ID?",
        "I am setting up out of band management on the FlexPod what is the subnet?",
        "I am setting up out of band management on the FlexPod what is the gateway?"
    ]

    output_file = "model_output.txt"
    # Clear the file first
    with open(output_file, "w") as file:
        file.write("")

    test_model("Fine-Tuned Model", fine_tuned_model, tokenizer, questions, device, output_file)
    test_model_with_llm("Base Model", llm, questions, output_file)

def test_model(model_name, model, tokenizer, questions, device, output_file):
    with open(output_file, "a") as file:  # Open file in append mode
        file.write(f"\nTesting {model_name}:\n")
        print(f"\nTesting {model_name}:")
        for question in questions:
            answer = ask_model(question, model, tokenizer, device)
            output = f"Question: {question}\nAnswer: {answer}\n"
            file.write(output)
            print(output)

def test_model_with_llm(model_name, llm, questions, output_file):
    with open(output_file, "a") as file:  # Open file in append mode
        file.write(f"\nTesting {model_name}:\n")
        print(f"\nTesting {model_name}:")
        for question in questions:
            response = llm.invoke(question)
            output = f"{question}\n{response}\n"
            file.write(output)
            print(output)

def ask_model(question, model, tokenizer, device, max_length=128, num_beams=3):
    prompt = f"{question}\n "

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        with autocast():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if __name__ == "__main__":
    main()
