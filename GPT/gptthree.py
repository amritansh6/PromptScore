import openai
import os


class OpenAIGPT3:
    def __init__(self, prompts, example_prompt, api_key):
        self.prompts = prompts
        self.example_prompt = example_prompt
        self.api_key = api_key  # Get API key from environment variable
        self.engine = "gpt-4-turbo-preview"  # Update to the latest engine

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        openai.api_key = self.api_key

    def teach_model(self):
        try:
            prompts = self.prompts
            msg = "I am designing a prompt scoring system: Following is the criteria: NUMBER_OF_CONSTRAINT (Range 1 to 5) CONSTRAINT_COMPLEXITY Range(1 to 5) PROMPT_COMPLEXITY Range(1 to 5) Example Prompts and their score in the given order:"
            final_prompt_for_training = msg
            for prompt in prompts[4:]:
                final_prompt_for_training += f"{prompt[0]}{prompt[1]}{prompt[2]}{prompt[3]}"
            final_prompt_for_training += "Now based on the criteria give score for :" + self.example_prompt + "Output just the scores without explantion"

            response = openai.chat.completions.create(
                model=self.engine,
                messages=[{"role": "system", "content": final_prompt_for_training}]
            )
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_prompt_score(self, response):
        message_content = response.choices[0].message.content
        print(message_content)
        if len(message_content) ==3:
            digits = [int(char) for char in message_content]
            # Calculate the average of the digits
            average = sum(digits) / len(digits)
            return average

        return None

    def chat(self):
        # Implement chat functionality if needed
        pass
