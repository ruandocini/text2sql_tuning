#eval broken tables
# db_root_path='./dataset/bird_mod/databases/'
# data_mode='dev'
# diff_json_path='./dataset/results_broken_llama3_7b/dev.json'
# predicted_sql_path_kg='./dataset/results_broken_llama3_7b/'
# predicted_sql_path='./dataset/results_broken_llama3_7b/'
# ground_truth_path='./dataset/results_broken_llama3_7b/'
# num_cpus=16
# meta_time_out=30.0
# mode_gt='gt'
# mode_predict='gpt'

# Ta rolando pro normal, fazer funcionar pras modificações
# #eval normal tables
db_root_path='./data/bird/data/dev_databases_mod/'
data_mode='dev'
diff_json_path='./data/bird/dev.json'
# predicted_sql_path_kg='./dataset/results_finetuning/'
predicted_sql_path='./data/bird/data/dev_databases_mod/'
ground_truth_path='./data/bird/data/dev_databases_mod/'
num_cpus=16
meta_time_out=30.0
mode_gt='gt'
mode_predict='gpt'

# echo '''starting to compare with knowledge for ex'''
# python3 -u ./src/evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
# --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
# --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

echo '''starting to compare without knowledge for ex'''
python3 -u evaluation.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
--ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
--diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

# echo '''starting to compare with knowledge for ves'''
# python3 -u ./src/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path_kg} --data_mode ${data_mode} \
# --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
# --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}

# echo '''starting to compare without knowledge for ves'''
# python3 -u ./src/evaluation_ves.py --db_root_path ${db_root_path} --predicted_sql_path ${predicted_sql_path} --data_mode ${data_mode} \
# --ground_truth_path ${ground_truth_path} --num_cpus ${num_cpus} --mode_gt ${mode_gt} --mode_predict ${mode_predict} \
# --diff_json_path ${diff_json_path} --meta_time_out ${meta_time_out}