from openai import OpenAI

class OpenAIClient:
    def __init__(self, api_key):
        self.api_key = api_key

    def make_request(self, prompt):
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.001
        )
        return completion.choices[0].message.content

SIMPLE_PROMPT = """
You are supplied with the content of a specific column from a database and its current name.
This name is not representative, meaning it does not accurately describe the content of the column.
You are tasked with rephrasing the name of the column to better reflect its content.
Remember that this name should be simple and also descriptive.
The current column name is: "{column_name}"
The content of the column is as follows: "{content}"

Your reponse should come in the following format:
{{"repahrsed_column_name": "new_column_name"}}

And only that, nothing more is accepted.
"""