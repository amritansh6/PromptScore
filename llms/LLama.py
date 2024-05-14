from llamaapi import LlamaAPI
from matplotlib import pyplot as plt

from Evaluation.Evaluation_GPT import Evaluation_GPT
from LLMInterface import LLMInterface
from db.store_prompts import StoryPrompts


class LLama(LLMInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llama = LlamaAPI('LL-pSogkDynoNL7YsWcAr8o1jvUyeUjjG2pOvpwnTb7zdymyLbKLTEf147YevkbDuZI')

    def generate_story(self, prompt: str) -> str:
        api_request_json = {
            "model": "llama3-8b",
            "messages": [
                {"role": "system",
                 "content": prompt}
            ]
        }
        response = self.llama.run(api_request_json)
        print(response)
        return response.json()['choices'][0]['message']['content']


if __name__ == '__main__':
    llama = LLama('llama3')

    db_name_story = '../prompts_story.db'
    db1 = StoryPrompts(db_name_story)

    x_values = []
    y_values = []
    i = 0

    for prompt, final_score in db1.get_prompts():
        story = llama.generate_story(prompt)

        scores = []
        constraints = ["../Evaluation/Coherence.txt", "../Evaluation/Constraints.txt", "../Evaluation/Fluency.txt"]
        for constraint in constraints:
            evaluator = Evaluation_GPT(prompt, story, "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", constraint)
            try:
                score = int(evaluator.evaluate_constraints())
            except Exception as e:
                print(f"An error occurred: {e}")
                score = 3  # taking average score
            scores.append(score)
        print(scores)
        x_values.append(i)
        y_values.append(sum(scores)/len(scores))
        i += 1

    plt.plot(x_values, y_values)
    plt.show()

    db1.close()
