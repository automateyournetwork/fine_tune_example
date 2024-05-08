import json

def transform_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            # Assuming 'prompt' is a sequence of questions and 'chosen' contains the corresponding answers
            messages = []
            # You can adjust this initial role message to suit the context of your training
            messages.append({"role": "system", "content": "You are a computer networking expert specializing in network routing tables."})
            
            # Split the prompt into separate questions (assuming each question is separated by two newlines)
            questions = data['prompt'].split('\n\n')
            answers = data['chosen'].split('\n\n')
            
            for question, answer in zip(questions, answers):
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})

            # Wrap transformed messages in a new dictionary with the key 'messages'
            json.dump({"messages": messages}, outfile)
            outfile.write('\n')  # Write a newline to separate each entry in the JSONL output

input_path = 'training_dataset.jsonl'
output_path = 'chatgpt_format_training_dataset.jsonl'
transform_data(input_path, output_path)
