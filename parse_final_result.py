import json


#read json
with open('predictions/Gemma2-9B-it/predictions_normal.json') as f:
    data = json.load(f)

# print(data["1"].split("CREATED SQL: ")[1].split("END OF QUESTION")[0])

parsed_data = {}

error_count = 0
for idx, line in enumerate(data.values()):
    try:
        # print(line)
        # print(line.split("start_sql")[2].split("end_sql")[0])
        db = line.split("\n\t----- bird -----\t")[-1]
        query = line.split("start_sql")[2].split("end_sql")[0].replace(
            f"\n\t----- bird -----\t{db}","").replace(
                "\r\n", ""
            )
        final = f"{query}\n\t----- bird -----\t{db}"
        print(final)
        parsed_data.update({idx:final})
    except:
        db = line.split("\n\t----- bird -----\t")[-1]
        # query = line.split("start_sql")[2].split("end_sql")[0]
        query = "SELECT * FROM table"
        final = f"{query}\n\t----- bird -----\t{db}"
        parsed_data.update({idx:final})
        # print(error_count)
        error_count += 1
    
    
with open("parsed_result.json", "w") as f:
    json.dump(parsed_data, f, indent=4)

print(f"Total errors: {error_count}")
raise Exception()

replacement = "\n\t"
replacecer = "\r\n\n\t"

for idx, row in enumerate(data.values()):

    final = ""
    
    if 'CREATED SQL: ' in row:
        query = row.split('CREATED SQL: ')[1].split('END OF QUESTION')[0]

        if "\n\t----- bird -----\t" in row:
            query = query.split("\n\t----- bird -----\t")[0]

        db = row.split("\n\t----- bird -----\t")[-1]
        final = f"{query}\n\t----- bird -----\t{db}"
        # print(f"""{final}""")
        print(query)
    else:
        db = row.split("\n\t----- bird -----\t")[-1]
        final = f"SELECT * FROM table\n\t----- bird -----\t{db}"
    
    

        