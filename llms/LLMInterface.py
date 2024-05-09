
class LLMInterface:
    def generate_story(self, prompt: str) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")
