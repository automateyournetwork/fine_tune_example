import csv
import json
import random
from langchain_community.llms import Ollama

# Function to generate a question from the detailed FlexPod information
def generate_question(llm, context, question_template):
    prompt = f"The following VLAN information is from the Cisco Validated Design FlexPod Datacenter with Generative AI Inferencing Design and Deployment Guide and the following information is taken from the guide:\n{context}\n\nGenerate a question similar to this related to the VLAN:\n{question_template}. Generate a factual response and nothing else. Keep your answers short and to the point."
    response = llm.invoke(prompt)
    return response.strip()

# Function to generate the chosen (correct) answer
def generate_chosen_response(question, llm):
    chosen_prompt = f"This data is being used to fine-tune an LLM. Based on the given information, please provide the chosen answer for this question: {question}"
    response = llm.invoke(chosen_prompt)
    print(f"Chosen response: {response.strip()}")
    return response.strip()

# Function to generate the rejected (incorrect) answer
def generate_rejected_response(chosen_response, llm):
    rejected_prompt = f"Here is the chosen (correct) response:\n{chosen_response}\nCould you alter it slightly to be incorrect?"
    response = llm.invoke(rejected_prompt)
    print(f"Rejected response: {response.strip()}")
    return response.strip()

# Function to create and save the dataset as JSONL
def save_dataset_as_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')
    print(f"Saved {len(data)} entries to '{filename}'")

# Function to generate dataset with 50 entries per row of the CSV
def generate_dataset(llm, csv_file):
    dataset = []

    question_templates = [
        "What is the <VLAN Name> VLAN ID?",
        "What is the VLAN ID <VLAN ID>'s subnet?",
        "What is the <VLAN Name> gateway?",
        "What is the purpose of the <VLAN Name>?",
        "What is the purpose of VLAN ID <VLAN ID>?",
        "What is the subnet for <VLAN Name>?",
        "What is the gateway for VLAN ID <VLAN ID>?"
    ]

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            context = f"VLAN ID - {row['VLAN ID']}, Name - {row['Name']}, Usage - {row['Usage']}, IP Subnet - {row['IP Subnet']},IP Gateway - {row['IP Gateway']}"

            # Generate 50 prompts per row
            for _ in range(50):
                template = random.choice(question_templates)
                question_template = template.replace("<VLAN Name>", row['Name']).replace("<VLAN ID>", row['VLAN ID'])
                question = generate_question(llm, context, question_template)
                chosen_answer = generate_chosen_response(question, llm)
                rejected_answer = generate_rejected_response(chosen_answer, llm)
                dataset.append({
                    'prompt': question,
                    'chosen': chosen_answer,
                    'rejected': rejected_answer
                })

    # Save the dataset to a JSONL file
    save_dataset_as_jsonl(dataset, 'manual_training_dataset.jsonl')

# Initialize the Llama model with the specified model name
llm = Ollama(model="llama3")

# Generate the dataset from the provided CSV file
generate_dataset(llm, 'training_dataset.csv')
