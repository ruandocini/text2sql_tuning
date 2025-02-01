from openai import OpenAI
from langchain_ollama.llms import OllamaLLM


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
    def __init__(self, model_name="llama3.2:1b", logger=None):
        self.model_name = model_name
        self.client = OllamaLLM(model=model_name, temperature=0.001)
        self.logger = logger

    def make_request(self, prompt):
        return self.client.invoke(
            prompt,
        )


SIMPLE_PROMPT_COLUMN = """
You are supplied with the content of a specific column from a database and its current name.
This name is not representative, meaning it does not accurately describe the content of the column.
You are tasked with rephrasing the name of the column to better reflect its content.
Remember that this name should be simple and also descriptive.
The current column name is: "{column_name}"
The content of the column is as follows: "{content}"

Your reponse should come in the following format:
{{"rephrased_column_name": "new_column_name"}}

The new name must be a contiguos string. No spaces or special characters in it.

And only that, nothing more is accepted.
Only generate one new name for per column.
It is obligatory to responde with a json object. And only that.
Respect the json format.
A json is the answer all the times.
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
