import argparse
import csv

import torch

from GPT.generatePrompts import GeneratePrompts
from data.createDataLoader import getDataset
from db import prompts_db
from db.prompts_db import PromptsDb
from GPT.gptthree import OpenAIGPT3
from transformers import BertTokenizer, AutoModel, AutoTokenizer

from db.store_prompts import StoryPrompts
from models.bert import BertRegressor
from models.llama2 import LlamaRegressor
from trainer import Trainer

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

    generatePrompts = GeneratePrompts(prompt, "sk-proj-jzjcetv6c5nhrjhLEQ9dT3BlbkFJTKTRbNgcCBsqBhPIQ65u",
                                      '/Users/amritanshmishra/PycharmProjects/PromptScore/GPT/Instruction.txt')
    instructions = generatePrompts.load_instructions()
    message = generatePrompts.call_gpt4_api(instructions)
    message_array = message.split('\n\n')
    for message in message_array:
        print(message)

    db_name_story = 'prompts_story.db'
    db1 = StoryPrompts(db_name_story)
    # for prompt in message_array:
    #     gpt3=OpenAIGPT3(all_prompts,prompt,'sk-proj-jzjcetv6c5nhrjhLEQ9dT3BlbkFJTKTRbNgcCBsqBhPIQ65u')
    #     response=gpt3.teach_model()
    #     prompt_score=gpt3.get_prompt_score(response)
    #     print(prompt_score)
    #     db1.add_prompt(prompt, prompt_score)

    # print(message)

    db.close()
    db1.close()
