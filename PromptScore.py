import csv
from db import prompts_db
from db.prompts_db import PromptsDb


class CsvImporter:
    def __init__(self, csv_file):
        self.csv_file = csv_file

    def import_prompts(self, db):
        with open(self.csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                prompt, number_constraints,constraint_complexity,prompt_complexity = row
                db.add_prompt(prompt, number_constraints, constraint_complexity, prompt_complexity)

if __name__ == '__main__':
    db_name = 'prompts.db'
    csv_file = 'your_prompts_file.csv'

    db = PromptsDb(db_name)
    importer = CsvImporter(csv_file)
    importer.import_prompts(db)

    all_prompts = db.get_prompts()
    print(all_prompts)

    db.close()