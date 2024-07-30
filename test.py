import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LlamaForCausalLM
from datasets import load_dataset

def fixture():
    return """"
    train_example
    "Based on the database schema below and the question, create a SQL query that will return the desired result:
    DATABASE SCHEMA
    ---------------------
    CREATE TABLE records (
    id number,
    nutrient_absorption number,
    systolic_blood_pressure_sbp number,
    active_metabolic number,
    hepatic_disease number,
    surgery number,
    diastolic_blood_pressure_dbp number,
    gastrointestinal_disease number,
    organ_failure number,
    );
    ---------------------
    QUESTION:  subject states that he / she has current hepatic disease.
    """

torch.cuda.empty_cache()

peft_model_id = "ruandocini/llama31-8b-lora-sql"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# batch = tokenizer(fixture(), return_tensors='pt')

data = load_dataset("csv", data_files={"train":["bird_dev.csv"]})
data = data.map(lambda samples: tokenizer(samples['train_example']), batched=True)
data = data["train"][['input_ids', 'attention_mask']]


output_tokens = model.generate(**data, max_new_tokens=100)
print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))

# with torch.cuda.amp.autocast():
#   output_tokens = model.generate(**batch, max_new_tokens=100)

# print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
