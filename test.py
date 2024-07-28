import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, LlamaForCausalLM

def fixture():
    return """"
    "Based on the database schema below and the question, create a SQL query that will return the desired result:
    DATABASE SCHEMA
    ---------------------
    CREATE TABLE nuclear_power_plants (
    Id text,
    Name text,
    Latitude text,
    Longitude text,
    Country text,
    Status text,
    ReactorType text,
    ReactorModel text,
    ConstructionStartAt text,
    OperationalFrom text,
    OperationalTo text,
    Capacity text,
    LastUpdatedAt text,
    Source text,
    );
    ---------------------
    QUESTION: What are the top 10 countries with most number of operational plants?
    CREATED SQL: 
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

batch = tokenizer(fixture(), return_tensors='pt')

with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=100)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
