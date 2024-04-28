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
                messages=[{"role": "system", "content": prompt_for_gpt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)

    def evaluate_constraints(self):
        instructions = self.load_instructions(self.file_path)
        return self.call_gpt4_api(instructions)


if __name__ == '__main__':
    evaluator = Evaluation_GPT("Explore, in exactly 250 words, the psychological toll of amnesia on the protagonist as they piece together their past while grappling with their current predicament aboard the massive ship. Must include the words: psychological, amnesia, and predicament. Utilize a stream-of-consciousness narrative style for this exploration.",
                               "The ship's hum is a constant reminder of my unmoored reality, vast as the sea it cuts through. Every corridor and door seems familiar, yet alien, my steps echo with the emptiness of my memories. Amnesia – the word is a cold, clinical label that doesn’t begin to capture the sheer psychological disorientation, the vertigo that comes from not knowing who you are. They tell me my name, they smile with warmth I should probably recognize, but it's like looking at a photograph where you don’t belong, though everyone insists you do.The dining hall’s clatter might as well be the cacophony of a foreign market, the faces around me masks that change features with each blink. I try piecing together past moments, grasping at the slivers that surface—laughter, anger, sorrow—ephemeral and uncatchable like wisps of smoke. This predicament is a maze with no visible exit, each turn looped back to the same confusion, the same frustration. How can I plan a future when the past is a puzzle whose pieces don’t fit?Nightmares haunt me; or are they memories? They slip through my fingers at dawn, leaving a residue of dread. There’s a fear that perhaps I’m not the hero of my story, maybe a villain, or worse, an inconsequential extra. This ship carries me forward, yet I'm tethered to the unknown, each wave a mocking taunt: who are you? With each undulating wave, the question pounds harder, resonant with the ship's own heartbeat, indistinguishable from my own.",
                               "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","coh_detailes.txt")

    print(evaluator.evaluate_constraints())