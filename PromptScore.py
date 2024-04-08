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
    #db.delete_all_prompts()

    db = PromptsDb(db_name)
    db.delete_all_prompts()
    importer = CsvImporter(csv_file)
    importer.import_prompts(db)

    all_prompts = db.get_prompts()
    #print(all_prompts[1])
    gpt3 = OpenAIGPT3(all_prompts, "Write a story about bhopal", "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")
    response=gpt3.teach_model()
    #response=gpt3.score_prompt("Write a story which contains a dragon")
    print(response)

    db.close()
