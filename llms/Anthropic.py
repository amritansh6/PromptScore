import anthropic
from matplotlib import pyplot as plt

from Evaluation.Evaluation_GPT import Evaluation_GPT
from db.store_prompts import StoryPrompts
from llms.LLMInterface import LLMInterface


class Anthropic(LLMInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(
            api_key="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",
        )

    def generate_story(self, prompt: str) -> str:
        message = self.client.messages.create(
            model="claude-3-opus-20240229",
            system="Generate a creative story based on a given prompt",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text


if __name__ == '__main__':
    claud = Anthropic('claude-3')
    # claud.generate_story('Write a story about a man born in Bhopal')

    db_name_story = '../prompts_story.db'
    db1 = StoryPrompts(db_name_story)

    x_values = []
    y_values = []
    i = 1

    for prompt, final_score in db1.get_prompts():
        story = claud.generate_story(prompt)

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

        print(story)
        print(scores)
        x_values.append(i)
        y_values.append(sum(scores) / len(scores))
        i += 1

    plt.xticks(x_values)
    plt.plot(x_values, y_values)
    plt.show()

    db1.close()
