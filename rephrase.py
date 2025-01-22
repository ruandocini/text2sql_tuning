import json
import sqlite3
from llms import OpenAIClient, SIMPLE_PROMPT

class Rephrase:
    def __init__(self, database_dir: str, schema: dict):
        self.database_dir = database_dir
        self.schema = schema

    def rephrase_column_names(self, use_all_columns: bool = False, top_k_rows: int = 200) -> str:

        mapping = {}

        for database in self.schema.keys():
            with sqlite3.connect(self.database_dir) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {database}")
                database_content = cursor.fetchall()
                openai = OpenAIClient("API_KEY")
                for idx, column in enumerate(self.schema[database]):
                    column_sample = [content[idx] for content in database_content][0:top_k_rows]
                    rephrased_column = openai.make_request(SIMPLE_PROMPT.format(column_name=column, content=column_sample))
                    mapping.update({column: rephrased_column})
                    print(f"Ran algorithm for {idx} of {len(self.schema[database])} columns.")

        return mapping

if __name__ == "__main__":
    database_dir = "data/bird/data/dev_databases/california_schools/california_schools.sqlite"

    with open("mapper_of_columns.json", "r") as file:
        mapper_of_columns = file.read()
        mapper_of_columns = json.loads(mapper_of_columns)

    extract = mapper_of_columns["california_schools"]["columns"]["frpm"]

    schema = {
        "frpm": list(extract.values())[0:10]
    }

    original_schema = {
        "frpm": list(extract.keys())[0:10]
    }
    
    rephrase = Rephrase(database_dir, schema)
    rephrased_column = rephrase.rephrase_column_names()

    rephrased_schema = {
        "frpm": list(rephrased_column.values())
    }

    for idx, column in enumerate(original_schema["frpm"]):
        print(f"Original column name: {column}")
        print(f"Rephrased column name: {rephrased_schema['frpm'][idx]}")
        print("")