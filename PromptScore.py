import argparse
import csv
import json

import matplotlib.pyplot as plt
from transformers import BertTokenizer

from GPT.generatePrompts import GeneratePrompts
from GPT.gptthree import OpenAIGPT3
from db.prompts_db import PromptsDb
from db.store_prompts import StoryPrompts
from keys import Keys

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class CsvImporter:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def import_prompts(self, db):
        with open(self.csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                prompt, number_constraints, constraint_complexity, clarity, prompt_complexity = row
                db.add_prompt(prompt, number_constraints, constraint_complexity, prompt_complexity)

    def import_prompts_for_score(self, csv_file):
        prompts = []
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                prompt, number_constraints, constraint_complexity, clarity, prompt_complexity = row
                prompts.append(prompt)
        return prompts


def prepare_data_for_bert(inputs):
    encoding = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return encoding


def encode_data(data):
    inputs = []
    labels = []
    for entry in data:
        prompt_text, score1, score2, score3 = entry
        inputs.append(prompt_text.strip())
        labels.append([score1, score2, score3])

    return inputs, labels


def plot_graph(db1):
    x_values = range(1, 11)
    y_values = []
    for prompt, final_score in db1.get_prompts():
        y_values.append(final_score)

    print("len(y_values) ", len(y_values))
    plt.plot(x_values, y_values)
    plt.show()


if __name__ == '__main__':
    db_name = 'prompts.db'
    csv_file = 'Prompts_Amritansh.csv'
    parser = argparse.ArgumentParser(description='PromptScore')

    db = PromptsDb(db_name)
    db.delete_all_prompts()
    importer = CsvImporter(csv_file)
    importer.import_prompts(db)
    prompts = importer.import_prompts_for_score('Prompts_Manish.csv')

    all_prompts = db.get_prompts()
    all_prompts = all_prompts[2:]
    prompt_scores = []
    prompt = "Write a story about a man born in Bangalore"

    generatePrompts = GeneratePrompts(prompt, Keys.GPT_KEY,
                                      './GPT/Instruction.txt')
    instructions = generatePrompts.load_instructions()
    message = generatePrompts.call_gpt4_api(instructions)
    message = message[7:-3]  # removing ```json chars
    message_array = []

    with open("prompts.json", "w") as file:
        file.write(message)

    with open("prompts.json", "r") as file:
        data = json.load(file)

    for prompt in data:
        message_array.append(prompt["prompt"])

    db_name_story = 'prompts_story.db'
    db1 = StoryPrompts(db_name_story)
    for prompt in message_array:
        gpt3 = OpenAIGPT3(all_prompts, prompt, Keys.GPT_KEY)
        response = gpt3.teach_model()
        prompt_score = gpt3.get_prompt_score(response)
        # print(prompt, prompt_score)
        db1.add_prompt(prompt, prompt_score)

    plot_graph(db1)

    db.close()
    db1.close()
