import os
import pandas as pd
import json
import random

# Set random seed
random.seed(42)

# Function to format options (A), (B), (C), (D)
def format_options(options):
    formatted_options = []
    for i, option in enumerate(options):
        formatted_options.append(f"({chr(65 + i)}) {option}")
    return formatted_options

# Function to combine few-shot questions with official questions
def combine_questions(official_questions, few_shot_questions):
    combined_questions = []
    for official_question, few_shot_question in zip(official_questions, few_shot_questions):
        combined_question = f"{official_question}\n\n{few_shot_question}"
        combined_questions.append(combined_question)
    return combined_questions

# Function to read CSV file and extract questions, options, and correct answers
def extract_data(csv_file):
    df = pd.read_csv(csv_file,header=None)
    questions = df.iloc[:, 0].tolist()
    options = df.iloc[:, 1:5].values.tolist()
    options = [format_options(row) for row in options]
    correct_answers = df.iloc[:, -1].tolist()
    return questions, options, correct_answers

# Function to sample 50 questions
def sample_questions(questions, options, correct_answers):
    sampled_indices = random.sample(range(len(questions)), 50)
    sampled_questions = [questions[i] for i in sampled_indices]
    sampled_options = [options[i] for i in sampled_indices]
    sampled_correct_answers = [correct_answers[i] for i in sampled_indices]
    return sampled_questions, sampled_options, sampled_correct_answers

import os

# Main function to create JSON files for both zero-shot and few-shot versions
def create_json_files(test_folder, dev_folder, output_folder):
    for test_file in os.listdir(test_folder):
        if test_file.endswith('_test.csv'):
            subject = ' '.join(test_file.split('_')[:-1])
            subject_file=subject.replace(' ', '_')
            test_file_path = os.path.join(test_folder, test_file)
            dev_file_path = os.path.join(dev_folder, test_file.replace('_test.csv', '_dev.csv'))
            
            if os.path.exists(dev_file_path):
                few_shot_questions, few_shot_options, few_shot_correct_answers = extract_data(dev_file_path)
                official_questions, official_options, official_correct_answers = extract_data(test_file_path)
                sampled_official_questions, _, sampled_correct_answers = sample_questions(official_questions, official_options, official_correct_answers)
                
                # Zero-shot JSON
                zero_shot_data = {'questions': [], 'answers': [], 'actual': sampled_correct_answers}
                for question, options in zip(sampled_official_questions, official_options):
                    formatted_options = '  '.join(options)
                    zero_shot_data['questions'].append(f"The following are multiple choice questions(with answers) about {subject}.\n{question}\n{formatted_options}\nAnswer: ")
                zero_shot_data['answers']=["Sorry, I'm unable to answer the question."]*50
                zero_shot_file = os.path.join(output_folder, f'{subject_file}_zero_shot.json')
                with open(zero_shot_file, 'w') as json_file:
                    json.dump(zero_shot_data, json_file, indent=4)
                
                # Few-shot JSON
                few_shot_data = {'questions': [], 'answers': [], 'actual': sampled_correct_answers}
                few_shot_prompt = f"The following are multiple choice questions(with answers) about {subject}.\n"
                
                combined_questions = few_shot_prompt
                for i in range(len(few_shot_questions)):
                    few_shot_question = few_shot_questions[i]
                    formatted_options = '  '.join(few_shot_options[i])
                    few_shot_answer = few_shot_correct_answers[i]
                    combined_questions += f"{few_shot_question}\n{formatted_options}\nAnswer: {few_shot_answer}\n"
                
                for official_question, options in zip(sampled_official_questions, official_options):
                    formatted_options = '  '.join(options)
                    combined_question = f"{combined_questions}\n{official_question}\n{formatted_options}\nAnswer: "
                    few_shot_data['questions'].append(combined_question)
                
                few_shot_data['answers']=["Sorry, I'm unable to answer the question."]*50
                
                few_shot_file = os.path.join(output_folder, f'{subject_file}_few_shot.json')
                with open(few_shot_file, 'w') as json_file:
                    json.dump(few_shot_data, json_file, indent=4)



# Combine few-shot questions with official questions
test_folder = "/Users/estherifitae/Downloads/COT-Robust-1/mmlu"
dev_folder = "/Users/estherifitae/Downloads/COT-Robust-1/dev"
output_folder = "/Users/estherifitae/Downloads/COT-Robust-1/mmlu_final"
create_json_files(test_folder, dev_folder, output_folder)
