import sqlite3


class PromptsDb:

    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        """
        This function is used to create the table for storing sample prompts for few shot learning along with there,
        prompt scores annotated by humans.
        """
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS prompts
                                  (id INTEGER PRIMARY KEY, prompt TEXT, NUMBER_CONSTRAINTS INTEGER, CONSTRAINT_COMPLEXITY INTEGER, PROMPT_COMPLEXITY INTEGER)''')
        self.conn.commit()

    def add_prompt(self, prompt, NUMBER_CONSTRAINTS, PROMPT_COMPLEXITY, CONSTRAINT_COMPLEXITY):
        """
        This function is used to add prompts to the db.
        :param prompt: TEXT
        :param NUMBER_CONSTRAINTS: Score for Number of Constraint
        :param PROMPT_COMPLEXITY: Score for Prompt Complexity.
        :param CONSTRAINT_COMPLEXITY: Score for Constraint Complexity.
        :return:
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO prompts (prompt, NUMBER_CONSTRAINTS,CONSTRAINT_COMPLEXITY,PROMPT_COMPLEXITY) VALUES (?, ?)",
            (prompt, NUMBER_CONSTRAINTS, CONSTRAINT_COMPLEXITY, PROMPT_COMPLEXITY))
        self.conn.commit()

    def get_prompts(self):
        """
        Retrieve all prompts and their scores from the database.

        Returns:
        - list of tuples: A list of prompts and their scores.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT prompt,NUMBER_CONSTRAINTS,CONSTRAINT_COMPLEXITY,PROMPT_COMPLEXITY FROM prompts")
        prompts = cursor.fetchall()
        return prompts

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()

