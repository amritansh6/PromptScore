from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
import torch


def encode_data(row):
    tokenizer = get_tokenizer()
    prompt = f"""###INPUT: Given the following prompt, score the prompt on the basis of four metrics (1 to 5) - #constraints, Constraint Complexity, Clarity, Prompt Complexity.
              ###PROMPT: {row['Prompt']}
              ###SCORES:"""

    output = f"""###SCORES: {{#constraints: {row['#constraints']},
              'Constraint Complexity': {row['Constraint Complexity']},
              'Clarity': {row['Clarity']},
              'Prompt Complexity': {row['Prompt Complexity']}
              }}"""

    tokens = tokenizer(prompt,
                       truncation=True,
                       max_length=256,
                       padding="max_length")

    output_tokens = tokenizer(output,
                              truncation=True,
                              max_length=256,
                              padding="max_length")

    tokens["labels"] = output_tokens['input_ids'].copy()
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

        args=TrainingArguments(
            output_dir="./training",
            remove_unused_columns=False,
            per_device_train_batch_size=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            max_steps=200,
            learning_rate=2.5e-5,
            logging_steps=5,
            fp16=True,
            optim="paged_adamw_8bit",
            save_strategy="steps",
            save_steps=50,
            report_to="none",
        ))

    trainer.train()


def test(model, prompt):
    txt = f"""###INPUT: Given the following prompt, score the prompt on the basis of four metrics (1 to 5) - #constraints,Constraint Complexity,Clarity,Prompt Complexity.
              ###PROMPT: {prompt}
              """
    tokens = tokenizer(txt, return_tensors="pt")['input_ids'].to("cuda")
    op = model.generate(tokens, max_new_tokens=200)
    print(tokenizer.decode(op[0]))


if __name__ == "__main__":
    csv_file_path = 'Prompts_Manish.csv'
    tokenizer = get_tokenizer()
    model = get_model()
    dataset = prepare_data(csv_file_path, tokenizer)
    train(model, dataset, tokenizer)

    test_prompt = "Write a story where the ticking clock on the wall begins to unravel along with the fabric of time itself. The story must contain exactly 200 words and be written in reverse chronological order."
    test(model, prompt=test_prompt)
