import torch
from torch import nn
from transformers import BertModel, AutoModelForCausalLM, BitsAndBytesConfig


class LlamaRegressor(nn.Module):
    def __init__(self):
        super(LlamaRegressor, self).__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        self.llama = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                                                 device_map="auto",
                                                 quantization_config=bnb_config)
        self.mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        attention_mask = torch.zeros_like(input_ids)
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        last_token_representation = last_hidden_state[:, -1, :]
        scores = self.mlp(last_token_representation)
        #scores = self.mlp(pooled_output)
        return scores
