from openai import OpenAI
from langchain_ollama.llms import OllamaLLM
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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
    def __init__(self, model_name="meta-llama/Llama-3.2-3B", mode="auto"):
        device = "cuda:0" if torch.cuda.is_available() else "mps"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_name), 
            return_dict=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = device
        self.model.to(device)
        
    def make_request(self, prompt):
        final_input = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )
         
        final_input = {k: v.to(self.device) for k, v in final_input.items()}
        raw_outputs = self.model.generate(**final_input, max_new_tokens=300)
        raw_outputs = raw_outputs.to(self.device)
        decoded_outputs = self.tokenizer.batch_decode(
            raw_outputs, skip_special_tokens=True
        )
        return decoded_outputs
