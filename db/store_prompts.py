import sqlite3


class StoryPrompts:

    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        """
        This function is used to create the table for storing sample prompts for few shot learning along with there,
        prompt scores annotated by humans.
        """
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS story_prompts
                                  (id INTEGER PRIMARY KEY, prompt TEXT, FINAL_SCORE INTEGER)''')
        self.conn.commit()

    def add_prompt(self, prompt, FINAL_SCORE):
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
            "INSERT INTO story_prompts (prompt, FINAL_SCORE) VALUES (?, ?)",
            (prompt, FINAL_SCORE))
        self.conn.commit()

    def get_prompts(self):
        """
        Retrieve all prompts and their scores from the database.

        Returns:
        - list of tuples: A list of prompts and their scores.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT prompt,FINAL_SCORE FROM story_prompts")
        prompts = cursor.fetchall()
        return prompts

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()

    def delete_all_prompts(self):
        """
        Delete all prompts from the database.
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM story_prompts")
        self.conn.commit()