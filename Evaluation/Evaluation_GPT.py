import openai


class Evaluation_GPT:
    def __init__(self, prompt, story, api_key, file_path):
        self.prompt = prompt
        self.story = story
        self.api_key = api_key
        self.engine = "gpt-4-turbo-preview"
        self.file_path = file_path

        openai.api_key = self.api_key

    def load_instructions(self, file_path):
        with open(file_path, 'r') as file:
            instructions = file.read()
        return instructions

    def call_gpt4_api(self, instructions):
        prompt_for_gpt = f"{instructions}\n\nPrompt:\n{self.prompt}\n\nStory:\n{self.story}\n\nEvaluation Form (scores ONLY):\n- Constraints:"
        try:
            response = openai.chat.completions.create(
                model=self.engine,
                messages=[{"role": "system", "content": prompt_for_gpt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)

    def evaluate_constraints(self):
        instructions = self.load_instructions(self.file_path)
        return self.call_gpt4_api(instructions)

def load_prompt(file_path):
    with open(file_path, 'r') as file:
        instructions = file.read()
    return instructions

if __name__ == '__main__':
    prompt=load_prompt("prompt.txt")
    story=load_prompt("story.txt")
    #print(story)
    evaluator = Evaluation_GPT(prompt, story,"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","coh_detailes.txt")
    print(evaluator.evaluate_constraints())