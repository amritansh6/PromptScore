import argparse
import csv

import torch

from data.createDataLoader import getDataset
from db import prompts_db
from db.prompts_db import PromptsDb
from GPT.gptthree import OpenAIGPT3
from transformers import BertTokenizer

from models.bert import BertRegressor
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

    all_prompts = db.get_prompts()
    all_prompts = all_prompts[2:]
    inputs, labels = encode_data(all_prompts)
    input_to_bert = prepare_data_for_bert(inputs)

    model = BertRegressor()
    dataLoader = getDataset()
    train_loader, val_loader = dataLoader.getDataLoader(input_to_bert, labels)
    trainer = Trainer(model, train_loader, val_loader, torch.nn.MSELoss())

    trainer.load_checkpoint('checkpoint/best_model.pt')
    prompt="Write a story about a boy in about 500 words"
    encoding=tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = encoding['input_ids']
    print(trainer.evaluate_prompt(input_ids))

    # #print(all_prompts[1])
    gpt3 = OpenAIGPT3(all_prompts, "Write a story about a boy in about 500 words", "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")
    response=gpt3.teach_model()
    print(response)
    #response=gpt3.score_prompt("Write a story about a boy in about 500 words")
    prompt_score=gpt3.get_prompt_score(response)
    # print(prompt_score)

    db.close()
