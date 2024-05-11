import argparse
import csv

from PromptScore import tokenizer
from db.prompts_db import PromptsDb


class Experiment:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def import_prompts(self, db):
        with open(self.csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                prompt, number_constraints, constraint_complexity, clarity, prompt_complexity = row
                db.add_prompt(prompt, number_constraints, constraint_complexity, prompt_complexity)


if __name__ == '__main__':
    db_name = 'prompts.db'
    csv_file = 'Prompts_Amritansh.csv'
    parser = argparse.ArgumentParser(description='PromptScore')

    db = PromptsDb(db_name)
    db.delete_all_prompts()
    importer = Experiment(csv_file)
    importer.import_prompts(db)

    all_prompts = db.get_prompts()
    all_prompts = all_prompts[2:]



    db.close()
