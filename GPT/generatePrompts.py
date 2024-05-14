import openai


class GeneratePrompts:
    def __init__(self, prompt, api_key, file_path):
        self.prompt = prompt
        self.api_key = api_key
        self.engine = "gpt-4-turbo-preview"
        self.file_path = file_path

        openai.api_key = self.api_key

    def load_instructions(self):
        with open(self.file_path, 'r') as file:
            instructions = file.read()
        return instructions

    def call_gpt4_api(self, instructions):
        prompt_for_gpt = f"{instructions}\n\nStartingPrompt:\n{self.prompt}\n\n Output just the json file"
        try:
            response = openai.chat.completions.create(
                model=self.engine,
                messages=[{"role": "system", "content": prompt_for_gpt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)



