import argparse
import csv
from db import prompts_db
from db.prompts_db import PromptsDb
from GPT.gptthree import OpenAIGPT3


class CsvImporter:
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
    parser.add_argument('')
    #db.delete_all_prompts()

    db = PromptsDb(db_name)
    db.delete_all_prompts()
    importer = CsvImporter(csv_file)
    importer.import_prompts(db)

    all_prompts = db.get_prompts()
    #print(all_prompts[1])
    gpt3 = OpenAIGPT3(all_prompts, "Develop a science fiction story set on a distant planet inhabited by sentient alien beings, where a human astronaut crash-lands and must form an unlikely alliance with the native creatures to survive and find a way back home.", "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")
    response=gpt3.teach_model()
    #response=gpt3.score_prompt("Write a story which contains a dragon")
    print(response)

    db.close()
