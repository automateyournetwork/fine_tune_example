import csv
import json
import random
from langchain_community.llms import Ollama

# Function to generate the chosen (correct) answer
def generate_chosen_response(question, llm, context):
    chosen_prompt = f"This data is being used to fine-tune an LLM. Based on the given information, please provide the chosen answer for this question: {question}\n\nContext: {context}\n\nRespond with the answer only."
    response = llm.invoke(chosen_prompt)
    print(f"Chosen: {response}")
    return response.strip()

# Function to generate the rejected (incorrect) answer
def generate_rejected_response(chosen_response, llm):
    rejected_prompt = f"Here is the chosen (correct) response:\n{chosen_response}\nCould you alter it slightly to be incorrect? Respond with the answer only."
    response = llm.invoke(rejected_prompt)
    print(f"Rejected: { response }")
    return response.strip()

# Function to create and save the dataset as JSONL
def save_dataset_as_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')
    print(f"Saved {len(data)} entries to '{filename}'")

# Function to generate dataset with a specified number of entries per row of the CSV
def generate_dataset(llm, csv_file, entries_per_row=25):
    dataset = []

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

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            context = f"VLAN ID - {row['VLAN ID']}, Name - {row['Name']}, Usage - {row['Usage']}"
            if row['IP Subnet'] != "N/A":
                context += f", IP Subnet - {row['IP Subnet']}"
            if row['IP Gateway'] != "N/A":
                context += f", IP Gateway - {row['IP Gateway']}"

            template = random.choice(question_templates)
            question = template.replace("<VLAN Name>", row['Name']).replace("<VLAN ID>", row['VLAN ID'])
            chosen_answer = generate_chosen_response(question, llm, context)
            rejected_answer = generate_rejected_response(chosen_answer, llm)
            dataset.append({
                'prompt': question,
                'chosen': f"Answer: {chosen_answer}",
                'rejected': f"Answer: {rejected_answer}"
            })
    return dataset

# Main function to generate datasets with two different models
def main():
    csv_file = 'training_dataset.csv'

    # Initialize the Llama model with the specified model names
    llm_mistral = Ollama(model="mistral")
    llm_llama3 = Ollama(model="llama3")

    # Generate dataset with Mistral model
    dataset_mistral = generate_dataset(llm_mistral, csv_file)
    save_dataset_as_jsonl(dataset_mistral, 'training_dataset_mistral.jsonl')

    # Generate dataset with Llama3 model
    dataset_llama3 = generate_dataset(llm_llama3, csv_file)
    save_dataset_as_jsonl(dataset_llama3, 'training_dataset_llama3.jsonl')

if __name__ == "__main__":
    main()
