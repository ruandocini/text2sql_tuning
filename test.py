import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

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

model = AutoModel.from_pretrained("ruandocini/llama31-8b-lora-sql") 
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

batch = tokenizer(fixture(), return_tensors='pt')

with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=100)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))