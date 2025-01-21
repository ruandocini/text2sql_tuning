from distutils.dir_util import copy_tree
import os
import json
import sqlite3


def json_loader(filename:str):
    with open(filename) as f:
        data = json.load(f)
        return data
    
def alter_table_names():
    for db in dev_databases:
        db_specfic_mapper = mapper[db]["tables"]
        conn = sqlite3.connect(f"unified/bird/data/dev_databases_mod/{db}/{db}.sqlite")
        # conn = sqlite3.connect(f"/unified/bird/data/dev_databases_mod/debit_card_specializing/debit_card_specializing.sqlite")
        cursor = conn.cursor()
        for table in db_specfic_mapper.keys():
            cursor.execute(f"ALTER TABLE \"{table}\" RENAME TO {db_specfic_mapper[table]}")
        
        # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            cursor.execute(f"SELECT * FROM {db_specfic_mapper[table]}")
            tables = cursor.fetchall()
            # print(f"SELECT * FROM {db_specfic_mapper[table]}")
            # print(tables)

def alter_columns_names():
    for db in dev_databases:
        db_specfic_mapper = mapper[db]["tables"]
        conn = sqlite3.connect(f"unified/bird/data/dev_databases_mod/{db}/{db}.sqlite")
        cursor = conn.cursor()
        # print(db)
        for table in db_specfic_mapper.keys():

            table_mapper = mapper[db]["columns"][table]
            # print(table_mapper)

            cursor.execute(f"PRAGMA table_info(\"{table}\");")
            columns = cursor.fetchall()

            transaction = f"PRAGMA foreign_keys=off;"
            cursor.execute(transaction)

            transaction2 = "\n\nBEGIN TRANSACTION;"
            cursor.execute(transaction2)
            
            transaction3 = f"\n\nALTER TABLE \"{table}\" RENAME TO \"{table}_old\";\n\n"
            cursor.execute(transaction3)

            transaction4 = f"CREATE TABLE \"{table}\"\n(\n"

            renaming = [
                f"\"{table_mapper[column[1]]}\" {column[2]}\n"
                for column in columns
            ]
            renaming = ",".join(renaming)

            transaction4 += renaming
            transaction4 += ");"
            cursor.execute(transaction4)

            inserting_tb_name = [
                f"\"{table_mapper[column[1]]}\""
                for column in columns
            ]

            old_colums_names = [
                f"\"{column[1]}\""
                for column in columns
            ]

            old_colums_names_stringfied = ", ".join(old_colums_names)


            inserting_tb_name = ", ".join(inserting_tb_name)
            inserting_tb_name = f"INSERT INTO \"{table}\" ({inserting_tb_name})\n" + f"SELECT {old_colums_names_stringfied}\n" + f"FROM \"{table}_old\";\n\n"
            # print(inserting_tb_name)
            cursor.execute(inserting_tb_name)
            # print("\n")

            # cursor.execute(f"PRAGMA table_info(\"{table}\");")
            # columns = cursor.fetchall()
            # print(columns)

            cursor.execute("COMMIT;\n")
            cursor.execute("PRAGMA foreign_keys=on;\n")

            cursor.execute(f"SELECT * FROM \"{table}\"")
            columns = cursor.fetchall()
            # print(columns)

copy_tree("unified/bird/data/dev_databases", "unified/bird/data/dev_databases_mod")

dev_databases = os.listdir("unified/bird/data/dev_databases_mod")

ignored_files = [".DS_Store", "dev.json", "dev_gold.sql", "predict_dev.json", "predict_dev__.json", 
                 "predict_dev_perfect.json", "predict_dev_cols.json", "dev_gold_cols.sql"
                 ]

[
    dev_databases.remove(file) if file in dev_databases else None
    for file in ignored_files
]



mapper = json_loader("mapper_of_columns.json")

original_dev_answers = json_loader("unified/bird/data/dev_databases/dev.json")

alter_columns = False
alter_tables = True

if alter_columns:
    print("Modifying columns")
    alter_columns_names()

    # original_dev_answers = json_loader("unified/bird/data/dev_databases_mod/predict_dev_cols.json")
    # original_dev_answers = [
    #     {
    #         "SQL": question.split("\n\t----- bird -----\t")[0],
    #         "db_id": question.split("\n\t----- bird -----\t")[1]
    #     }
    #     for question in original_dev_answers.values()
    # ]

if alter_tables:
    print("Modifying tables")
    alter_table_names()

    # for question in original_dev_answers:
    #     local_mapper = mapper[question["db_id"]]
    #     for starting_tb, finishing_tb in local_mapper["tables"].items():
    #         question["SQL"] = question["SQL"].replace(f"FROM {starting_tb} ", f"FROM {finishing_tb} ")
    #         question["SQL"] = question["SQL"].replace(f"JOIN {starting_tb} ", f"JOIN {finishing_tb} ")
    #         question["SQL"] = question["SQL"].replace(f"FROM {starting_tb}\n\t", f"FROM {finishing_tb}\n\t")
    #         question["SQL"] = question["SQL"].replace(f"JOIN `{starting_tb}`", f"JOIN `{finishing_tb}`")
    #         question["SQL"] = question["SQL"].replace(f"FROM `{starting_tb}`", f"FROM `{finishing_tb}`")
    #         question["SQL"] = question["SQL"].replace(f"FROM {starting_tb}", f"FROM {finishing_tb}")

# with open("unified/bird/data/dev_databases_mod/dev.json", "w") as f:
#     json.dump(original_dev_answers, f, indent=4)

#create SQL file only with the SQL queries
# accumulator_ground_truth = {}
# with open("unified/bird/data/dev_databases_mod/dev_gold.sql", "w") as f:
#     for idx, question in enumerate(original_dev_answers):
#         f.write(f"{question['SQL']}\t{question['db_id']}\n")
#         accumulator_ground_truth.update({str(idx):f"{question['SQL']}\n\t----- bird -----\t{question['db_id']}"})

# with open("unified/bird/data/dev_databases_mod/predict_dev_perfect.json", "w") as f:
#     json.dump(accumulator_ground_truth, f, indent=4)