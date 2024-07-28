import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def fixture():
    return """"
    "Based on the database schema below and the question, create a SQL query that will return the desired result:
    DATABASE SCHEMA
    ---------------------
    CREATE TABLE records (
    id number,
    sinusal_rhythm number,
    severe_sepsis number,
    hemodynamic_instability number,
    social_security number,
    septic_shock number,
    age number,
    );
    ---------------------
    QUESTION: patient over 18 years admitted in intensive care unit and having severe sepsis criteria or septic shock documented or suspected define as bone criteria
    CREATED SQL:
    """

torch.cuda.empty_cache()

peft_model_id = "ruandocini/llama31-8b-lora-sql"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
model.resize_token_embeddings(len(tokenizer))

# Load the Lora model
model = PeftModel.from_pretrained(model=model, model_id=peft_model_id, config=config)

batch = tokenizer(fixture(), return_tensors='pt')

with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=100)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
