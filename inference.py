import json
import pandas as pd
import argparse
from llms import OllamaClient, HuggingFaceClient
import os
import time
import func_timeout
from concurrent.futures import ThreadPoolExecutor

class Inference():
    def __init__(self, full_data_path:str, model:str, mode:str, framework:str):
        self.full_data_path = full_data_path
        self.model = model
        self.mode = mode
        self.framework = framework

    def run_inference(self):
        data = pd.read_csv(f"{self.full_data_path}", sep=",")
        llm ={
            "ollama": OllamaClient(model_name=self.model, temperature=0.001, max_new_tokens=400),
            "huggingface": HuggingFaceClient(model_name=self.model, mode="auto")
        }[self.framework]

        obrigatory_markings = "\n\t----- bird -----\t"

        inference_json = {}

        def process_data(i):
            print(f"Processing {i} of {len(data)}")

            response_model_prompt = """
            /n
            EXAMPLE JSON OUTPUT:

            {
                "question": "Which is the highest mountain in the world?",
                "sql_answer": "SELECT name FROM mountains WHERE height = (SELECT MAX(height) FROM mountains);"
            }

            Use the above JSON output to generate the SQL query for the given question.
            { "question": 
            """

            created_sql = None
            try:
                created_sql = func_timeout.func_timeout(60, llm.make_request, args=(data.iloc[i]["train_example"] + response_model_prompt,))
            except func_timeout.FunctionTimedOut:
                print("Timeout occurred")
                created_sql = None

            if created_sql is None:
                final_str = " " + obrigatory_markings + db
            else:
                db = data.iloc[i]["db_id"]
                final_str = created_sql.replace("sql_start", "").replace("sql_end", "").replace("```", "").replace("sql", "") + obrigatory_markings + db
            
            inference_json[i] = final_str
            print(final_str)
            db = data.iloc[i]["db_id"]

        inference_json = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            executor.map(process_data, range(len(data)))

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
    args = parser.parse_args()

    inference = Inference(args.full_data_path, args.model, args.mode, args.framework)

    inference.save_inference(inference.run_inference())