from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch
from torch import nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class LlamaRegressor(nn.Module):
    def __init__(self):
        super(LlamaRegressor, self).__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        llama = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                                                     device_map="auto",
                                                     quantization_config=bnb_config,
                                                     output_hidden_states=True)
        if torch.cuda.device_count() > 1:
            llama.is_parallelizable = True

        llama.model_parallel = True
        llama.gradient_checkpointing_enable()
        llama = prepare_model_for_kbit_training(llama)

        peft_config = LoraConfig(inference_mode=False,
                                 r=8,
                                 lora_alpha=32,
                                 lora_dropout=0.1,
                                 peft_type=TaskType.QUESTION_ANS)

        self.llama = get_peft_model(llama, peft_config)
        self.mlp = nn.Sequential(
            nn.Linear(2048, 512).to(dtype=torch.float32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3).to(dtype=torch.float32)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        attention_mask = torch.zeros_like(input_ids)
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.hidden_states[-1]
        last_token_representation = last_hidden_state[:, -1, :]
        scores = self.mlp(last_token_representation)
        return scores
