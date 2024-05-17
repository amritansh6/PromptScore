from matplotlib import pyplot as plt

from Evaluation.Evaluation_GPT import Evaluation_GPT
from db.store_prompts import StoryPrompts
from keys import Keys
from llms.Alpaca import Alpaca
from llms.Anthropic import Anthropic
from llms.Gemma import Gemma
from llms.LLama import LLama

if __name__ == '__main__':
    llama = LLama('llama3')
    alpaca = Alpaca('alpaca-7b')
    claud = Anthropic('claude-3')
    gemma = Gemma('gemma-7b')

    models = [alpaca, claud, gemma]
    constraints = ["Coherence", "Constraints", "Fluency"]

    db_name_story = '../prompts_story.db'
    db1 = StoryPrompts(db_name_story)
    prompts_map = {}
    for prompt, final_score in db1.get_prompts():
        prompts_map[prompt] = final_score
    sorted_prompt = {k: round(v, 2) for k, v in sorted(prompts_map.items(), key=lambda item: item[1])}

    x_values = [score for prompt, score in sorted_prompt.items()]
    y_values = []

    for constraint in constraints:
        constrain_file_path = "../Evaluation/" + constraint + ".txt"
        scores = []

        for model in models:
            model_scores = []
            for prompt, final_score in sorted_prompt.items():
                story = model.generate_story(prompt)
                evaluator = Evaluation_GPT(prompt, story, Keys.GPT_KEY,
                                           constrain_file_path)
                try:
                    score = int(evaluator.evaluate_constraints())
                except Exception as e:
                    print(f"An error occurred: {e}")
                    score = 3  # taking average score
                model_scores.append(score)
            scores.append(model_scores)
            print("model processed: ", model.model_name)

        for score_, model_ in zip(scores, models):
            plt.plot(x_values, score_, label=model_.model_name, marker='o')

        print(constraint, scores)

        plt.title(constraint)
        plt.xticks(x_values)
        plt.legend()
        plt.show()

    db1.close()


"""
[alpaca, claud, gemma]
Coherence [[5, 5, 5, 5, 4, 4, 4, 4, 2, 4], [5, 5, 5, 5, 4, 4, 1, 2, 1, 2], [5, 5, 5, 4, 4, 3, 2, 2, 1, 1]]
Constraints [[5, 4, 4, 2, 3, 1, 1, 3, 1, 1], [4, 5, 5, 3, 4, 1, 1, 1, 1, 1], [2, 5, 4, 2, 1, 1, 1, 1, 1, 1]]


Coherence [[5, 5, 5, 4, 5, 4, 1, 1, 2, 4], [5, 5, 5, 5, 5, 4, 2, 1, 4, 3], [5, 5, 5, 4, 4, 3, 1, 1, 2, 1]]
Constraints [[1, 5, 5, 2, 4, 4, 5, 1, 2, 5], [5, 5, 4, 2, 2, 3, 1, 1, 1, 1], [4, 5, 5, 2, 2, 1, 1, 1, 1, 1]]
Fluency [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]
"""
