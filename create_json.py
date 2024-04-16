import json
import random
random.seed(42)

# Load the original JSON data
with open('/home/users/nus/e1154542/scratch/COT-Robust/temporal.json', 'r') as f:
    old_data = json.load(f)

# Read the contents of the text file
with open('/home/users/nus/e1154542/scratch/COT-Robust/cot_temporal.txt', 'r') as file:
    text_content = file.read().strip()
# print(old_data)
data=random.sample(old_data['examples'],50)

# print(data)

# Extract the "input" values
questions = [text_content +'\n\n'+ example['input']+ "\nA: Let's think step by step." for example in data]

# Create a new JSON structure
new_data = {
    "question": questions,
    "answer": []
}

# Write the new JSON data to a file
with open('/home/users/nus/e1154542/scratch/COT-Robust/data/50_temporal.json', 'w') as f:
    json.dump(new_data, f, indent=4)


answers = [example['target'] for example in data]
print(answers)
