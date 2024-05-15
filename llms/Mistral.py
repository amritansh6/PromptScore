import matplotlib.pyplot as plt
from llamaapi import LlamaAPI

from Evaluation.Evaluation_GPT import Evaluation_GPT
from LLMInterface import LLMInterface
from db.store_prompts import StoryPrompts


class Mistral(LLMInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llama = LlamaAPI('LL-WIZiIHXM9Q7hJrAdIukwRCIT1XZmbZSAv4I1OqZbECJZOrNyf1q2aaPcwcKpCWCe')

    def generate_story(self, prompt: str) -> str:
        api_request_json = {
            "model": "mistral-7b-instruct",
            "messages": [
                {"role": "system",
                 "content": prompt}
            ]
        }
        response = self.llama.run(api_request_json)
        return response.json()['choices'][0]['message']['content']


if __name__ == '__main__':
    llama = Mistral('mistral-7b-instruct')
    # llama.generate_story('Write a story about a man born in Bhopal')

    db_name_story = '../prompts_story.db'
    db1 = StoryPrompts(db_name_story)

    x_values = []
    y_values = []
    i = 1

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
        y_values.append(sum(scores) / len(scores))
        i += 1

    plt.xticks(x_values)
    plt.plot(x_values, y_values)
    plt.show()

    db1.close()
