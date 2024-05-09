from llamaapi import LlamaAPI

from LLMInterface import LLMInterface


class Gemma(LLMInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llama = LlamaAPI('LL-WIZiIHXM9Q7hJrAdIukwRCIT1XZmbZSAv4I1OqZbECJZOrNyf1q2aaPcwcKpCWCe')

    def generate_story(self, prompt: str) -> str:
        api_request_json = {
            "model": "gemma-7b",
            "messages": [
                {"role": "system",
                 "content": prompt}
            ]
        }
        response = self.llama.run(api_request_json)
        print(response.json())

if __name__ == '__main__':
    llama=Gemma('gemma-7b')
    llama.generate_story('Write a story about a man born in Bhopal')