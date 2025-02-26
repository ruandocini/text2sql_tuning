import json
from typing import List
import pandas as pd
import argparse
from llms import OllamaClient, HuggingFaceClient
import os
import time
import func_timeout

class Inference():
    def __init__(self, full_data_path:str, model:str, mode:str, framework:str, batch_size:int):
        self.full_data_path = full_data_path
        self.model = model
        self.mode = mode
        self.framework = framework
        self.batch_size = batch_size

    def run_inference(self):
        data = pd.read_csv(f"{self.full_data_path}", sep=",")
        llm ={
            "ollama": OllamaClient(model_name=self.model, temperature=0.001, max_new_tokens=400),
            "huggingface": HuggingFaceClient(model_name=self.model, mode="auto")
        }[self.framework]

        obrigatory_markings = "\n\t----- bird -----\t"

        inference_json = {}

        def process_data(idxs: List[int], data: pd.DataFrame):
            print(f"Processing {idxs[-1]} of {len(data)}")

            response_model_prompt = """
            Create the SQL query for the given question. Pay attention in the schema of the database that is supplied to you.
            It is crucial to the query creation.
            /n
            EXAMPLE JSON OUTPUT:

            {"think": "insert here the thought process of the query creation","__sql_answer": "SELECT columns FROM table;"}

            Use the above JSON output to generate the SQL query for the given question.
            {
            """

            local_inference = [
                data.iloc[idx]["train_example"] + response_model_prompt
                for idx in idxs
            ]

            start = time.time()
            created_sql = None
            try:
                created_sql = func_timeout.func_timeout(240, llm.make_request, args=(local_inference,))
                print(f"Time taken {time.time()-start}")
            except func_timeout.FunctionTimedOut:
                print("Timeout occurred")
                created_sql = None

            if created_sql is None:
                for idx in idxs:
                    db = data.iloc[idx]["db_id"]
                    final_str = " " + obrigatory_markings + db
                    inference_json[idx] = final_str
            else:
                for sql, idx in zip(created_sql, idxs):
                    db = data.iloc[idx]["db_id"]
                    try:
                        sql_fixed = json.loads("{" + sql.split('__sql_answer":"')[0].split('}')[0] + "}")["__sql_answer"]
                    except:
                        sql_fixed = ""
                    final_str =  sql_fixed + obrigatory_markings + db
                    inference_json[idx] = final_str
                    print(idx)
                    print(sql_fixed)

        inference_json = {}

        number_of_batches = (len(data) // self.batch_size) + 1

        for i in range(number_of_batches):
            batch = [j+self.batch_size*i for j in range(0, self.batch_size+1)]
            filtered_batch = []
            for ind in batch:
                if ind < len(data):
                    filtered_batch.append(ind)            
            
            if len(filtered_batch) > 0:
                process_data(
                    idxs=filtered_batch,
                    data=data
                )


        return inference_json
    
    def save_inference(self, inference_json):
        output_dir = f"predictions/{self.model}"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/inference_{self.mode}.json", "w") as f:
            json.dump(inference_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("full_data_path", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("mode", type=str)
    parser.add_argument("framework", type=str)
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()

    inference = Inference(args.full_data_path, args.model, args.mode, args.framework, args.batch_size)

    inference.save_inference(inference.run_inference())
