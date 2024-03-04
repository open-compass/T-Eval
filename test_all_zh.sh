echo "model_type: $1"

model_path=$2
echo "load model from: $model_path"

display_name=$3
echo "Model display name: $display_name"

if [ -z "$4" ]; then
    meta_template="nan"
else
    meta_template=$4
fi
echo "Model meta_template: $meta_template"

echo "evaluating instruct ..."
python test.py --model_type $1 --resume --out_name instruct_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/instruct_v2_zh.json --eval instruct --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating review ..."
python test.py --model_type $1 --resume --out_name review_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/review_str_v2_zh.json --eval review --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating plan json ..."
python test.py --model_type $1 --resume --out_name plan_json_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/plan_json_v2_zh.json --eval plan --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating plan str ..."
python test.py --model_type $1 --resume --out_name plan_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/plan_str_v2_zh.json --eval plan --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating reason str ..."
python test.py --model_type $1 --resume --out_name reason_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/reason_str_v2_zh.json --eval reason --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating retrieve str ..."
python test.py --model_type $1 --resume --out_name retrieve_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/retrieve_str_v2_zh.json --eval retrieve --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating understand str ..."
python test.py --model_type $1 --resume --out_name understand_str_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/understand_str_v2_zh.json --eval understand --prompt_type str --model_path $model_path --model_display_name $display_name --meta_template $meta_template

echo "evaluating RRU (reason, retrieve, understand) json ..."
python test.py --model_type $1 --resume --out_name reason_retrieve_understand_json_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/reason_retrieve_understand_json_v2_zh.json --eval rru --prompt_type json --model_path $model_path --model_display_name $display_name --meta_template $meta_template