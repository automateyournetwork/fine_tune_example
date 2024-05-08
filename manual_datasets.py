import json
from langchain_community.llms import Ollama

# Function to generate a question from the detailed FlexPod information
def generate_question(llm, text):
    prompt = f"Generate a question based on this information: {text}"
    response = llm.invoke(prompt)
    return response

# Function to generate the chosen (correct) answer
def generate_chosen_response(question, llm):
    chosen_prompt = f"This data is being used to fine-tune an LLM. Based on the given information, please provide the chosen answer for this question: {question}"
    response = llm.invoke(chosen_prompt)
    print(f"Chosen response: { response}")
    return response

# Function to generate the rejected (incorrect) answer
def generate_rejected_response(question, llm):
    rejected_prompt = f"This data is being used to fine-tune an LLM. Based on the given information, please provide a plausible but incorrect answer for this question: {question}"
    response = llm.invoke(rejected_prompt)
    print(f"Rejected response: { response}")
    return response

# Function to create and save the dataset as JSONL
def save_dataset_as_jsonl(data, filename):
    with open(filename, 'w') as f:
        for entry in data:
            json.dump(entry, f)
            f.write('\n')
    print(f"Saved {len(data)} entries to '{filename}'")

# Function to generate dataset with 50 entries
def generate_dataset(llm, source_text):
    dataset = []
    for _ in range(50):  # Looping 50 times to create 50 entries
        question = generate_question(llm, source_text)
        chosen_answer = generate_chosen_response(question, llm)
        rejected_answer = generate_rejected_response(question, llm)
        dataset.append({
            'prompt': question,
            'chosen': chosen_answer,
            'rejected': rejected_answer
        })

    # Save the dataset to a JSONL file
    save_dataset_as_jsonl(dataset, 'manual_training_dataset.jsonl')

# Text containing the detailed description of FlexPod
flexpod_description = """
        Text you paste in fromy our PDF or whatever 
    """

# Initialize the Llama model with specified model name
llm = Ollama(model="phi3")

# Generate the dataset
generate_dataset(llm, flexpod_description)
