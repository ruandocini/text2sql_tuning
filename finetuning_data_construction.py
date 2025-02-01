import copy
import pandas as pd
import json
import os
from typing import Tuple, List
import typer

app = typer.Typer()


def questions_loader(specific_dataset: str) -> json:
    with open(f"data/{specific_dataset}/dev.jsonl") as f:
        data = [json.loads(line) for line in f]
    return data


def questions_loader_bird(specific_dataset: str) -> json:
    with open(f"data/dev.json") as f:
        data = json.load(f)
    return data


def tables_loader(specific_dataset: str) -> json:
    with open(f"data/tables.json") as f:
        data = json.load(f)
        # data = pd.DataFrame(data)
        return data


def mapper_loader() -> json:
    with open(f"mapper_of_columns.json") as f:
        data = json.load(f)
        # data = pd.DataFrame(data)
        return data


def table_reconstructor(
    columns: list,
    relevant_index: int,
    data_types: list,
    primary_keys: list,
    foreign_keys: list,
    tables: list,
) -> str:
    primary_key = ""
    foreign_key = ""
    foreign_key_mapper = {
        foreign_key_position: reference_key
        for foreign_key_position, reference_key in foreign_keys
    }
    column_absolute_index_mapper = {
        table_idx: table_name[1]
        for table_idx, table_name in enumerate(columns)
    }
    table_absolute_index_mapper = {
        table_idx: table_name[0]
        for table_idx, table_name in enumerate(columns)
    }
    table_name_mapper = {
        table_idx: table_name for table_idx, table_name in enumerate(tables)
    }
    final_create_columns_statements = ""
    for absolute_list_index, (table_idx, column) in enumerate(columns):
        if table_idx == relevant_index:
            create_table = f"CREATE TABLE {table_name_mapper[table_idx]} (\n"
            column_creation = f"{column} {data_types[absolute_list_index]}"
            if absolute_list_index in primary_keys:
                primary_key = "PRIMARY KEY " + "(" + column + "),\n"
            if absolute_list_index in foreign_key_mapper.keys():
                reference_to_table = table_name_mapper[
                    table_absolute_index_mapper[
                        foreign_key_mapper[absolute_list_index]
                    ]
                ]
                foreign_key += (
                    "FOREIGN KEY "
                    + "("
                    + column
                    + f") REFERENCES {reference_to_table}({column_absolute_index_mapper[foreign_key_mapper[absolute_list_index]]}),\n"
                )
            final_create_columns_statements += column_creation + ",\n"
    final_statement = (
        create_table
        + final_create_columns_statements
        + primary_key
        + foreign_key
        + ");"
    )
    return final_statement


def database_reconstructor(table_descriptor: dict) -> str:

    final_db_statement = "DATABASE SCHEMA"
    final_db_statement += "\n---------------------\n"

    for idx, table in enumerate(table_descriptor["table_names_original"]):
        final_db_statement += table_reconstructor(
            columns=table_descriptor["column_names_original"],
            relevant_index=idx,
            data_types=table_descriptor["column_types"],
            primary_keys=table_descriptor["primary_keys"],
            foreign_keys=table_descriptor["foreign_keys"],
            tables=table_descriptor["table_names_original"],
        )
        final_db_statement += "\n"
    final_db_statement += "---------------------"
    # print(final_db_statement)
    return final_db_statement


def benchmark_reconstructor(tables: dict) -> dict:
    benchmark_databases = {}
    for table in tables:
        tables = database_reconstructor(table)
        benchmark_databases[table["db_id"]] = tables

    return benchmark_databases


def example_builder(
    databases: dict,
    database_id: str,
    question: str,
    generated_sql: str,
    evidence: str,
) -> str:
    db = (
        f"Based on the database schema below and the question, create a SQL query that will return the desired result:\n{databases[database_id]}\n"
        ""
    )
    question = f"QUESTION: {question.replace(evidence, "")}"
    generated_sql = f"CREATED SQL: {generated_sql}"
    file_end = "\nEND OF QUESTION"
    command = "\nGenerate just the SQL code starting it with 'start_sql' and ending the sql with 'end_sql' nothing else is allowed on the response. Do not add any explanations or comments."
    # print(db+question+command)
    return db + question + command


# + "\n" + generated_sql + file_end
# TODO ADICIONAR DNV QUANDO FOR FAZER O FINE TUNING


def dataset_builder(databases: dict, questions: list) -> str:
    return pd.DataFrame(
        [
            {
                "db_id": question["db_id"],
                "train_example": example_builder(
                    databases=databases,
                    database_id=question["db_id"],
                    question=question["question"],
                    generated_sql=question["query"],
                    evidence=question["evidence"],
                ),
            }
            for question in questions
        ]
    )


def map_tables(mapper: dict, database: dict) -> Tuple[dict, dict]:
    new_table_names = []
    mapper_position_table = {}
    for idx, table in enumerate(database["table_names_original"]):
        new_table_names.append(mapper[database["db_id"]]["tables"][table])
        mapper_position_table[idx] = table

    database["table_names_original"] = new_table_names
    return database, mapper_position_table


def map_columns(
    mapper: dict, mapper_position_table: dict, database: dict
) -> dict:
    new_columns_names = []
    new_columns_names.append(database["column_names_original"][0])
    for table in database["column_names_original"][1:]:
        position = [
            table[0],
            mapper[database["db_id"]]["columns"][
                mapper_position_table[table[0]]
            ][table[1]],
        ]
        new_columns_names.append(position)
    database["column_names_original"] = new_columns_names
    return database


def map_database(mapper: dict, database: str) -> dict:
    fake_db = copy.deepcopy(database)
    database, mapper_position_table = map_tables(mapper, fake_db)
    database = map_columns(mapper, mapper_position_table, database)
    return database


def map_benchmark(mapper: dict, tables: List[dict]) -> List[dict]:
    new_tables = []
    for database in tables:
        new_database = map_database(mapper, database)
        new_tables.append(new_database)
    return new_tables


def reconstruct(modification_type: list[str]):
    dataset = "bird"
    questions = questions_loader_bird(dataset)
    tables = tables_loader(dataset)

    mapper = mapper_loader()
    mapped_tables = map_benchmark(mapper, tables)

    if "columns" in modification_type:
        for idx, broken_columns in enumerate(mapped_tables):
            tables[idx]["column_names_original"] = broken_columns[
                "column_names_original"
            ]

    if "tables" in modification_type:
        for idx, broken_columns in enumerate(mapped_tables):
            tables[idx]["table_names_original"] = broken_columns[
                "table_names_original"
            ]

    databases = benchmark_reconstructor(tables)
    builded_dataset = dataset_builder(databases, questions)

    if not os.path.exists("data_modified"):
        os.makedirs("data_modified")

    if len(modification_type) >= 1:
        csv_name = "_".join(modification_type)
        csv_name = "broken_" + csv_name
    else:
        csv_name = ""

    print(csv_name)

    builded_dataset.to_csv(
        f"data_modified/{dataset}_{csv_name}.csv", index=False
    )


@app.command()
def reconstruct_default():
    reconstruct([])


@app.command()
def reconstruct_broken_columns_and_tables():
    reconstruct(["columns", "tables"])


@app.command()
def reconstruct_columns_broken():
    reconstruct(["columns"])


@app.command()
def reconstruct_tables_broken():
    reconstruct(["tables"])


if __name__ == "__main__":
    app()
