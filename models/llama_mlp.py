from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
import torch
import torch.nn as nn


class PromptScorerHead(nn.Module):
    def __init__(self):
        super(PromptScorerHead, self).__init__()
        self.mlp = nn.Linear(32000, 1)  # input dim = 32000, output dim = 1 (scores)

    def forward(self, x):
        x = self.mlp(x)
        return x


def encode_data(row):
    tokenizer = get_tokenizer()
    prompt = f"""{row['Prompt']}"""

    tokens = tokenizer(prompt,
                       truncation=True,
                       max_length=256,
                       padding="max_length")

    tokens["labels"] = torch.tensor([row['#constraints']])
    return tokens


def prepare_data(file_path, tokenizer):
    dataset = load_dataset('csv', data_files=file_path)
    dataset = dataset.map(encode_data)
    print(tokenizer.decode(dataset['train'][0]['input_ids']))
    print(dataset)
    dataset = dataset.remove_columns(
        ['Prompt', '#constraints', 'Constraint Complexity', 'Clarity', 'Prompt Complexity'])
    print(dataset)
    return dataset


def get_model():
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                                                 device_map="auto",
                                                 quantization_config=bnb_config)
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
                             peft_type=TaskType.QUESTION_ANS)
    model = get_peft_model(model, peft_config)
    model.scorer_head = PromptScorerHead()
    print(model.print_trainable_parameters())
    return model


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                                              padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def train(model, dataset, tokenizer):
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda pred: {"scores": model.scorer_head(pred.label_ids)},

        args=TrainingArguments(
            output_dir="./training",
            max_steps=10,
            learning_rate=2.5e-5,
            logging_steps=5,
            fp16=True,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=50,
        ))

    trainer.train()


def test(model, tokenizer, prompt):
    txt = f"""{prompt}"""
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to("cuda")

    model = model.to("cuda")  # Move the model to the CUDA device

    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
    logits = outputs.logits
    scores = model.scorer_head(logits[:, -1, :])  # get the scores from the classification head
    print(scores)


if __name__ == "__main__":
    csv_file_path = 'Prompts_Manish.csv'
    tokenizer = get_tokenizer()
    model = get_model()
    dataset = prepare_data(csv_file_path, tokenizer)
    train(model, dataset, tokenizer)

    test_prompt = "Write a story where the ticking clock on the wall begins to unravel along with the fabric of time " \
                  "itself. The story must contain exactly 200 words and be written in reverse chronological order. "
    test(model, tokenizer, prompt=test_prompt)
