import json
import sqlite3
from llms import (
    SIMPLE_PROMPT_COLUMN,
    SIMPLE_PROMPT_TABLE,
    OllamaClient,
    LLMClient,
)
from typing import List, Literal

import logging
import json
import sqlite3
from typing import List, Literal
import argparse

# Add this line to configure the default logger
logging.basicConfig(
    level=logging.INFO,
    format='\033[92m%(asctime)s\033[0m - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)


from llms import (
    SIMPLE_PROMPT_COLUMN,
    SIMPLE_PROMPT_TABLE,
    OllamaClient,
    LLMClient,
)


class Rephrase:
    def __init__(self, model: LLMClient):
        self.model = model

    def run_rephrasing(
        self,
        target: Literal["column", "table"],
        databases: List[str],
        mapper: dict,
        root_database_path: str,
        top_k_rows: int = 20,
    ) -> dict:

        if target == "column":

            logging.info(f"Starting rephrasing process for columns for {len(databases)} databases")

            for database in databases:
                logging.info(f"Rephrasing columns for {database} database")
                individual_database_mapper = {database: mapper[database]}
                mapper[database]["columns"] = (
                    self.rephrase_full_database_columns(
                        database=individual_database_mapper,
                        database_dir=f"{root_database_path}/{database}/{database}.sqlite",
                        top_k_rows=top_k_rows,
                    )[database]["columns"]
                )

            return mapper

        elif target == "table":
            # TODO
            pass

    def rephrase_full_database_columns(
        self, database: dict, database_dir: str, top_k_rows: int = 20
    ) -> str:
        for individual_databases in database.keys():
            extract = mapper_of_columns[individual_databases]["tables"]
            schema = {
                key: list(
                    mapper_of_columns[individual_databases]["columns"][
                        key
                    ].values()
                )
                for key in extract.keys()
            }
            rephrased_column = self.rephrase_column_names(
                database_dir=database_dir, schema=schema, top_k_rows=top_k_rows
            )
            for table in rephrased_column.keys():
                for column_original, column_mod in list(
                    mapper_of_columns[individual_databases]["columns"][
                        table
                    ].items()
                ):
                    mapper_of_columns[individual_databases]["columns"][table][
                        column_original
                    ] = rephrased_column[table][column_mod]


        return mapper_of_columns

    def rephrase_column_names(
        self,
        schema: dict,
        database_dir: str,
        use_all_columns: bool = False,
        top_k_rows: int = 20,
    ) -> str:

        mapping = {}

        counter = 0
        for table in schema.keys():
            mapping.update({table: {}})
            with sqlite3.connect(database_dir) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM '{table}'")
                database_content = cursor.fetchall()
                for idx, column in enumerate(schema[table]):
                    column_sample = [
                        content[idx] for content in database_content
                    ][0:top_k_rows]
                    rephrased_column = self.model.make_request(
                        SIMPLE_PROMPT_COLUMN.format(
                            column_name=column, content=column_sample
                        )
                    )
                    print("New column name: ", rephrased_column)
                    try:
                        # print(rephrased_column)
                        rephrased_column = json.loads(rephrased_column)[
                            "rephrased_column_name"
                        ]
                    except:
                        rephrased_column = column

                    mapping[table].update({column: rephrased_column+"_"+str(idx)})
                    # print(f"Ran algorithm for {idx} of {len(schema[table])} columns.")
            counter += 1
            logger.info(f"Rephrased {counter / len(schema.keys()) * 100:.2f}% of tables.")

        return mapping

    def rephrase_table_names(
        self,
        schema: dict,
        database_dir: str,
        use_specific_columns: List[str] = [],
        top_k_rows: int = 30,
    ) -> str:

        mapping = {}

        for col_pos, table in enumerate(schema.keys()):
            with sqlite3.connect(database_dir) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table}")
                database_content = cursor.fetchall()
                if len(use_specific_columns) == 0:
                    table_sample = database_content[0:top_k_rows]
                    rephrased_table = self.model.make_request(
                        SIMPLE_PROMPT_TABLE.format(
                            table_name=table, content=table_sample
                        )
                    )
                    try:
                        rephrased_table = json.loads(rephrased_table)[
                            "rephrased_table_name"
                        ]
                    except:
                        rephrased_table = table
                    mapping.update({table: rephrased_table+str(col_pos)})

        return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="LLM model name")
    args = parser.parse_args()

    logger.info("Loading database mapper")
    with open("mapper_of_columns.json", "r") as file:
        mapper_of_columns = file.read()
        mapper_of_columns = json.loads(mapper_of_columns)

    model = OllamaClient(args.model)
    rephrase = Rephrase(model)

    dev_databases = [
        "california_schools",
        "card_games",
        "codebase_community",
        "debit_card_specializing",
        "european_football_2",
        "financial",
        "formula_1",
        "student_club",
        "superhero",
        "thrombosis_prediction",
        "toxicology"
    ]

    new_mapper = rephrase.run_rephrasing(
        target="column",
        databases=dev_databases,
        mapper=mapper_of_columns,
        root_database_path="data/bird/data/dev_databases",
        top_k_rows=20,
    )

    with open(f"rephrased_mapper.json", "w") as file:
        json.dump(new_mapper, file, indent=4)

    # print("Mapper before")
    # print(mapper_of_columns["california_schools"]["columns"]["schools"])

    # california_schools = {
    #     "california_schools": mapper_of_columns["california_schools"]
    # }

    # mapper_of_columns["california_schools"]["columns"] = (
    #     rephrase.rephrase_full_database_columns(
    #         database=california_schools,
    #         database_dir=database_dir,
    #         top_k_rows=4,
    #     )["california_schools"]["columns"]
    # )

    # print("Mapper after")
    # print(mapper_of_columns["california_schools"]["columns"]["schools"])

    ## EXPERIMENTOS A SE FAZER
    ## 1. até quando o nome da coluna vai mudando com a inserção de mais colunas para o modelo?
    ## 2. Colocar barra de progresso para saber quanto tempo falta para terminar o processo para cada DB
    ##
