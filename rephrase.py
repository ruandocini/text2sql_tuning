import json
import sqlite3
from llms import SIMPLE_PROMPT_COLUMN, SIMPLE_PROMPT_TABLE, OllamaClient, LLMClient
from typing import List

class Rephrase:
    def __init__(self, model:LLMClient):
        self.model = model

    def rephrase_full_database_columns(self, database:dict, database_dir:str, top_k_rows: int = 20) -> str:
        for individual_databases in database.keys():
            extract = mapper_of_columns[individual_databases]["tables"]
            schema = {
                key: list(mapper_of_columns[individual_databases]["columns"][key].values()) for key in extract.keys()
            }
            rephrased_column = self.rephrase_column_names(database_dir=database_dir, schema=schema, top_k_rows=top_k_rows)
            for table in rephrased_column.keys():
                for column_original, column_mod in list(mapper_of_columns[individual_databases]["columns"][table].items()):
                    mapper_of_columns[individual_databases]["columns"][table][column_original] = rephrased_column[table][column_mod]

        return mapper_of_columns


    def rephrase_column_names(self, schema:dict, database_dir:str, use_all_columns: bool = False, top_k_rows: int = 20) -> str:

        mapping = {}

        for table in schema.keys():
            mapping.update({table: {}})
            with sqlite3.connect(database_dir) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table}")
                database_content = cursor.fetchall()
                for idx, column in enumerate(schema[table]):
                    column_sample = [content[idx] for content in database_content][0:top_k_rows]
                    rephrased_column = self.model.make_request(SIMPLE_PROMPT_COLUMN.format(column_name=column, content=column_sample))
                    try:
                        # print(rephrased_column)
                        rephrased_column = json.loads(rephrased_column)["rephrased_column_name"]
                    except:
                        rephrased_column = column

                    mapping[table].update({column: rephrased_column})
                    # print(f"Ran algorithm for {idx} of {len(schema[table])} columns.")
            print(f"Ran algorithm for {table}.")

        return mapping
    
    def rephrase_table_names(self, schema:dict, database_dir:str, use_specific_columns:List[str] = [], top_k_rows: int = 30) -> str:

        mapping = {}

        for table in schema.keys():
            with sqlite3.connect(database_dir) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table}")
                database_content = cursor.fetchall()
                if len(use_specific_columns) == 0:
                    table_sample = database_content[0:top_k_rows]
                    rephrased_table = self.model.make_request(SIMPLE_PROMPT_TABLE.format(table_name=table, content=table_sample))
                    try:
                        rephrased_table = json.loads(rephrased_table)["rephrased_table_name"]
                    except:
                        rephrased_table = table
                    mapping.update({table: rephrased_table})

        return mapping

if __name__ == "__main__":
    database_dir = "data/bird/data/dev_databases/california_schools/california_schools.sqlite"

    with open("mapper_of_columns.json", "r") as file:
        mapper_of_columns = file.read()
        mapper_of_columns = json.loads(mapper_of_columns)

    model = OllamaClient("llama3.2:1b")
    rephrase = Rephrase(model)

    print("Mapper before")
    print(mapper_of_columns["california_schools"]["columns"]["schools"])

    california_schools = {"california_schools":mapper_of_columns["california_schools"]}

    mapper_of_columns["california_schools"]["columns"] = rephrase.rephrase_full_database_columns(
        database=california_schools,
        database_dir=database_dir,
        top_k_rows=4
    )["california_schools"]["columns"]

    print("Mapper after")
    print(mapper_of_columns["california_schools"]["columns"]["schools"])


    # original_schema = {
    #     "frpm": list(extract.keys())[0:10]
    # }

    # rephrased_schema = {
    #     "frpm": list(rephrased_column.values())
    # }

    # for idx, column in enumerate(original_schema["frpm"]):
    #     print(f"Original column name: {column}")
    #     print(f"Rephrased column name: {rephrased_schema['frpm'][idx]}")
    #     print("")

    ## EXPERIMENTOS A SE FAZER 
    ## 1. até quando o nome da coluna vai mudando com a inserção de mais colunas para o modelo?
    ## 2. Colocar barra de progresso para saber quanto tempo falta para terminar o processo para cada DB
    ##