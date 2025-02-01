import sqlite3
import os
import pandas as pd
import json

"""Resposible for creating a mapper of the tables and columns so I can rename the database in a reproducible way"""


def get_database_information(db_path: str) -> dict:

    tables = read_json_file(db_path)

    mapper = {}

    for record in tables:
        old_to_new_names = {
            table: "table_" + str(idx)
            for idx, table in enumerate(record["table_names"])
        }
        mapper.update({record["db_id"]: old_to_new_names})

    return mapper


def non_meaningful_table_name(db_mapper: dict) -> dict:
    idx = 0
    table_mapping = {}
    for db in db_mapper:
        for tb_idx, table in enumerate(db_mapper[db]):
            table_mapping.update(
                {
                    str(idx): {
                        "database": db,
                        "table": table[0],
                        "non_meaningful_name": "table_" + str(tb_idx),
                    }
                }
            )
            idx += 1

    return table_mapping


def apply_table_modification(
    db_path: str, old_table_name: str, new_table_name: str
):

    if old_table_name != "sqlite_sequence":
        conn = sqlite3.connect(db_path)
        print(db_path)
        cursor = conn.cursor()

        mod_table_name = (
            f'ALTER TABLE "{old_table_name}" RENAME TO "{new_table_name}";'
        )
        # enable_mod = ' PRAGMA writable_schema=ON;'

        # print(mod_table_name)

        # questions = alter_sql_queries(
        #     questions=questions,
        #     db=table_mapping["database"],
        #     old_table_name=from_name,
        #     new_table_name=to_name
        # )

        # questions_train = alter_sql_queries(
        #     questions=questions_train,
        #     db=table_mapping["database"],
        #     old_table_name=from_name,
        #     new_table_name=to_name
        # )

        # list tables in database

        try:
            cursor.execute(mod_table_name)
            info = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
            print(info)
            conn.commit()
            conn.close()
        except Exception as e:
            print(e)
            conn.close()


def save_sql_changes(questions: list, saving_path: str):
    # print(json_questions_dev.replace('dev.json','dev.sql'))
    with open(saving_path, "w") as f:
        only_golden_sqls = [
            question["SQL"] + "\t" + question["db_id"]
            for question in questions
        ]
        # print(only_golden_sqls)
        f.write("\n".join(only_golden_sqls))
        f.close()


def read_json_file(json_file: str) -> dict:
    with open(json_file) as f:
        return json.load(f)


def alter_table_json(source_database: dict, saving_path: str):
    for table in source_database:
        table["table_names_original"] = [
            "table_" + str(idx) for idx in range(len(table["table_names"]))
        ]
        table["table_names"] = [
            "table_" + str(idx) for idx in range(len(table["table_names"]))
        ]
    with open(saving_path, "w") as f:
        json.dump(source_database, f, indent=4)


def alter_question_json(
    source_questions: dict, saving_path: str, table_mapping: dict
):
    # for record in table_mapping.to_dict(orient='records'):

    for question in source_questions:
        for old_table_name, new_table_name in table_mapping[
            question["db_id"]
        ].items():
            question["SQL"] = question["SQL"].replace(
                " " + old_table_name + " ", " " + new_table_name + " "
            )

    # save the change in the json file
    with open(saving_path, "w") as f:
        json.dump(source_questions, f, indent=4)


if __name__ == "__main__":
    # from distutils.dir_util import copy_tree, remove_tree
    # from_directory = "./dataset/bird/"
    # to_directory = "./dataset/bird_mod/"

    # #if directory exists, remove it
    # if os.path.exists(to_directory):
    #     remove_tree(to_directory)

    # copy_tree(from_directory, to_directory)

    # path = './dataset/bird_mod/'
    # #dev files
    # json_questions_dev = './dataset/bird_mod/dev/dev.json'
    # json_table_dev = './dataset/bird_mod/dev/dev_tables.json'
    # sql_dev = './dataset/bird_mod/dev/dev.sql'
    # #train files
    # json_questions_train = './dataset/bird_mod/train/train.json'

    json_table = "./data/tables.json"
    # json_table_dev = './dataset/bird_mod/dev/dev_tables.json'
    # sql_train = './dataset/bird_mod/train/train_gold.sql'
    # questions_file = "/Users/ruandocini/Documents/masters-v1/DAIL-SQL/dataset/process/BIRD-MOD-COLS-TEST_SQL_9-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096/dev.json"

    # questions = read_json_file(questions_file)

    consolidated_tables = read_json_file(json_table)

    mapper_of_columns = {}

    for database in consolidated_tables:
        # print(database["db_id"])
        mapper_of_columns[database["db_id"]] = {}
        mapper_of_columns[database["db_id"]]["columns"] = {}
        mapper_of_columns[database["db_id"]]["tables"] = {}
        for idx, table in enumerate(database["table_names_original"]):
            counter = 0
            mapper_of_columns[database["db_id"]]["columns"][table] = {}
            # print(table.upper())
            for column in database["column_names_original"]:
                if column[0] == idx:
                    # print(column[1])
                    mapper = {column[1]: "column_" + str(counter)}
                    mapper_of_columns[database["db_id"]]["columns"][table][
                        column[1]
                    ] = {}
                    mapper_of_columns[database["db_id"]]["columns"][table][
                        column[1]
                    ] = "column_" + str(counter)
                    counter += 1

            mapper_of_columns[database["db_id"]]["tables"][table] = (
                "table_" + str(idx)
            )

    # print(mapper_of_columns["movielens"])

    json.dump(mapper_of_columns, open("mapper_of_columns.json", "w"), indent=4)

    # print(questions["questions"][0]["prompt"])

    # read the json file
    # questions_dev = read_json_file(json_questions_dev)
    # table_dev = read_json_file(json_table_dev)

    # #read the json file
    # questions_train = read_json_file(json_questions_train)
    # table_train = read_json_file(json_table_train)

    # #see if there is a table_mapping.csv file and load it, delete it after loading
    # table_mapping_train = get_database_information(json_table_train)
    # table_mapping_dev = get_database_information(json_table_dev)

    # processed_bird_path = 'BIRD-MOD-TEST_SQL_9-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096/'
    # json.dump(table_mapping_train, open('./dataset/process/' + processed_bird_path + 'table_mapping_train.json', 'w'), indent=4)
    # json.dump(table_mapping_dev, open('./dataset/process/' + processed_bird_path + 'table_mapping_dev.json', 'w'), indent=4)

    # for database_name,table_mapper in table_mapping_train.items():
    #     for old_table_name,new_table_name in table_mapper.items():
    #         apply_table_modification(
    #             db_path=path + "databases/" + database_name + '/' + database_name + '.sqlite',
    #             old_table_name=old_table_name,
    #             new_table_name=new_table_name
    #         )

    # for database_name,table_mapper in table_mapping_dev.items():
    #     for old_table_name,new_table_name in table_mapper.items():
    #         apply_table_modification(
    #             db_path=path + "databases/" + database_name + '/' + database_name + '.sqlite',
    #             old_table_name=old_table_name,
    #             new_table_name=new_table_name
    #         )

    # alter_table_json(table_train, json_table_train)
    # alter_question_json(questions_train, json_questions_train, table_mapping_train)
    # save_sql_changes(questions_train, sql_train)

    # alter_table_json(table_dev, json_table_dev)
    # alter_question_json(questions_dev, json_questions_dev, table_mapping_dev)
    # save_sql_changes(questions_dev, sql_dev)

    # print("Created modified dataset in 'dataset/bird_mod/' directory.")
