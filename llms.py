from openai import OpenAI
from langchain_ollama.llms import OllamaLLM
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class LLMClient:
    def __init__(self, model_name):
        pass

    def make_request(self, prompt):
        pass


class OpenAIClient(LLMClient):
    def __init__(self, model_name="gpt-4"):
        self.client = OpenAI()
        self.model_name = model_name

    def make_request(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.001,
        )
        
        return completion.choices[0].message.content


class OllamaClient(LLMClient):
    def __init__(self, model_name="llama3.2:1b", logger=None, is_think_model=False, temperature=0.001, max_new_tokens=400):
        self.model_name = model_name
        self.client = OllamaLLM(model=model_name, temperature=0.001)
        self.logger = logger
        self.is_think_model = is_think_model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def make_request(self, prompt):
        response = self.client.invoke(
            prompt,
        )
        if "</think>" in response:
            response = response.split("</think>")[1]
            response = response.split("<think>")[0]
        
        response = response.strip().replace("\n", "")
    
        return response

    
class HuggingFaceClient(LLMClient):
    def __init__(self, model_name="meta-llama/Llama-3.2-3B", mode="auto", max_new_tokens=300):

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            return_dict=True,
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = device
        self.model.to(device)
        self.tokenizer.padding_side = "left"

        self.max_new_tokens = max_new_tokens
        
    def make_request(self, prompt):
        final_input = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
         
        raw_outputs = self.model.generate(**final_input, max_new_tokens=self.max_new_tokens)
        decoded_outputs = self.tokenizer.batch_decode(
            raw_outputs, skip_special_tokens=True
        )

        only_generated_ouput = [
            ind_decoded_output.replace(ind_prompt,"") for ind_prompt, ind_decoded_output in zip(prompt, decoded_outputs)
        ]
        return only_generated_ouput


SIMPLE_PROMPT_COLUMN = """
You are supplied with the content of a specific column from a database and its current name.
This name is not representative, meaning it does not accurately describe the content of the column.
You are tasked with rephrasing the name of the column to better reflect its content.
Remember that this name should be simple and also descriptive.
The current column name is: "{column_name}"
The content of the column is as follows: "{content}"
Your response should come in the following format:
{{"rephrased_column_name": "new_column_name"}}
The new name must be a contiguous string. No spaces or special characters in it.
And only that, nothing more is accepted.
Only generate one new name per column.
It is obligatory to respond with a JSON object. And only that.
Respect the JSON format.
A JSON has only one {{ in the beginning and one }} in the end.
"""
SIMPLE_PROMPT_TABLE = """
You are supplied with the content of a specific table from a database and its current name.
This name is not representative, meaning it does not accurately describe the content of the table.
You are tasked with rephrasing the name of the table to better reflect its content.
Remember that this name should be simple and also descriptive.
The current table name is: "{table_name}"
The content of the talbe is as follows: "{content}"
Your reponse should come in the following format:
{{"rephrased_table_name": "new_table_name"}}
And only that, nothing more is accepted.
"""