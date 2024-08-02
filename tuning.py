import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import json
import pandas as pd
from datasets import load_dataset


access_token = os.getenv("HUGGINGFACE_TOKEN")
os.environ["CUDA_VISIBLE_DEVICES"]="0"

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    load_in_8bit=True,
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    # target_modules=["q_proj", "v_proj"], #if you know the
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# import transformers
# from datasets import load_dataset
# data = load_dataset("Abirate/english_quotes")

# json_parsed = []

# with open('./dev.jsonl', 'r') as json_file:
#     json_list = list(json_file)
#     # print(json_list)

#     for json_str in json_list:
#       result = json.loads(json_str)
#       json_parsed.append(result)

for ds_number in range(0,9):
    data = load_dataset("csv", data_files={"train":[f"bird_train_{ds_number}.csv"]})
    data = data.map(lambda samples: tokenizer(samples['train_example']), batched=True)


    trainer = transformers.Trainer(
        model=model,
        train_dataset=data['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=3,
            per_device_eval_batch_size=3,
            gradient_accumulation_steps=3,
            warmup_steps=100,
            max_steps=1000,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir='outputs',
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

model.push_to_hub("ruandocini/llama31-8b-lora-sql2",
                  use_auth_token=True,
                  commit_message="basic training",
                  private=True)
