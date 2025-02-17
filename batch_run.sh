
# model="qwen2.5-coder:1.5b"

# osascript -e 'tell application "Terminal" to do script "ollama run \"'"$model"'\""'
# sleep 5
# # python inference.py "data_modified/bird_.csv" "$model" "default"
# # python inference.py "data_modified/bird_broken_columns_raw.csv" "$model" "broken_columns"
# python rephrase.py --model "$model"
# python finetuning_data_construction.py reconstruct-columns-broken --mapper "rephrased_mapper.json"
# python inference.py "data_modified/bird_broken_columns.csv" "$model" "rephrased_columns"
# sleep 1
# pkill -f "$model"

## VERSAO QUE RODOU HJ
model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# osascript -e 'tell application "Terminal" to do script "ollama run \"'"$model"'\""'
# sleep 1
# python inference.py "data_modified/bird_.csv" "$model" "default"
python3 inference.py "data_modified/bird_.csv" "$model" "default" "huggingface" 6
# python inference.py "data_modified/bird_broken_columns_raw.csv" "$model" "broken_columns"
python3 inference.py "data_modified/bird_broken_columns_raw.csv" "$model" "broken_columns" "huggingface" 6
# python rephrase.py --model "$model"
# python finetuning_data_construction.py reconstruct-columns-broken --mapper "rephrased_mapper.json"
# python inference.py "data_modified/bird_rephrased_columns_qwen2.5-code-14b.csv" "$model" "rephrased_columns"
python3 inference.py "data_modified/bird_rephrased_columns_qwen2.5-code-14b.csv" "$model" "rephrased_columns" "huggingface" 6
# sleep 1
# pkill -f "$model"

# # Running the evaluation from the default version of the dataset
# python database_modifier.py --mapper "rephrased_mapper.json"
# cp predictions/$model/inference_default.json data/bird/data/dev_databases_mod/predict_dev.json
# ./run_evaluation.sh "predictions/$model/acc/default.json"

# # Running the evaluation from the broken version of the dataset
# python database_modifier.py --mapper "mapper_of_columns.json" --alter_columns
# cp predictions/$model/inference_broken_columns.json data/bird/data/dev_databases_mod/predict_dev.json
# ./run_evaluation.sh "predictions/$model/acc/broken_columns.json"

# # Running the evaluation from the rephrased version of the dataset
# python database_modifier.py --mapper "rephrased_mapper.json" --alter_columns
# cp predictions/$model/inference_rephrased_columns.json data/bird/data/dev_databases_mod/predict_dev.json
# ./run_evaluation.sh "predictions/$model/acc/rephrased_columns.json"
