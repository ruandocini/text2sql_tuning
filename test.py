import time
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LlamaForCausalLM
from datasets import load_dataset
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser(description='What the program does')
parser.add_argument("file", type=str)

args = parser.parse_args()


torch.cuda.empty_cache()

# peft_model_id = "ruandocini/llama31-8b-lora-sql2"
# config = PeftConfig.from_pretrained(peft_model_id)
# model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model_name = "Gemma2-9B-it"
raw_model = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(raw_model)
model = AutoModelForCausalLM.from_pretrained(raw_model, return_dict=True, load_in_4bit=True, device_map='auto')

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load the Lora model
# model = PeftModel.from_pretrained(model, peft_model_id)

# batch = tokenizer(fixture(), return_tensors='pt')

data = pd.read_csv(f"dev/{args.file}.csv")
# final_input = tokenizer(data["train_example"].tolist(), return_tensors='pt', padding=True).to("cuda")
# raw_outputs = model.generate(**final_input, max_new_tokens=100)
# decoded_outputs = tokenizer.batch_decode(raw_outputs.detach().cpu().numpy(), skip_special_tokens=True)
# data = data.map(lambda samples: tokenizer(samples['train_example']), batched=True)
# data = data["train"][['input_ids', 'attention_mask']]

predictions = {}

batch_size = 7
for batch in range((len(data)//batch_size)+1):
    print(f"Batch {batch} of {(len(data)//batch_size)+1}")
    start = time.time()
    current_batch = data[batch*batch_size:(batch+1)*batch_size]
    final_input = tokenizer(current_batch["train_example"].tolist(), return_tensors='pt', padding=True).to("cuda")
    raw_outputs = model.generate(**final_input, max_new_tokens=100)
    decoded_outputs = tokenizer.batch_decode(raw_outputs.detach().cpu().numpy(), skip_special_tokens=True)
    print(decoded_outputs)
    # final_str = [output.split('CREATED SQL: ')[1].split('END OF QUESTION')[0] for output in decoded_outputs]
    final_str = [output for output in decoded_outputs]
    db = [line for line in current_batch["db_id"]]
    predictions.update({batch*batch_size+idx: f"{info[0]}\n\t----- bird -----\t{info[1]}" for idx, info in enumerate(zip(final_str,db))})
    # print({batch*batch_size+idx: f"{info[0]}\n\t----- bird -----\t{info[1]}" for idx, info in enumerate(zip(final_str,db))})
    print(f"Time taken: {time.time()-start}")
    with open(f"predictions/{model_name}/predictions_{args.file}.json", "w") as f:
        json.dump(predictions, f)

# predictions = {}
# for idx, example in enumerate(input_data):
#     print(f"Example {idx} of {len(input_data)}")
#     output_tokens = model.generate(**example, max_new_tokens=100)
#     output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
#     final_str = output.split('CREATED SQL: ')[1].split('END OF QUESTION')[0]
#     db = data["db_id"][idx]
#     # print(f"{final_str}+\n\t----- bird -----\t{db}")

#     predictions[idx] = f"{final_str}+\n\t----- bird -----\t{db}"


# output_tokens = model.generate(**data, max_new_tokens=100)
# print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

# with torch.cuda.amp.autocast():
#   output_tokens = model.generate(**batch, max_new_tokens=100)

# print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
