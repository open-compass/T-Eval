echo "model_type: $1"

if [ -z "$2" ]; then
    hf_path="."
else
    hf_path=$2
    echo "load model from: $hf_path"
fi

if [ -z "$3" ]; then
    display_name=$1
else
    display_name=$3
    echo "Model display name: $display_name"
fi

echo "evaluating instruct ..."
python test.py --model_type $1 --resume --out_name instruct_zh_$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/instruct_v1_zh.json --eval instruct --prompt_type json --hf_path $hf_path --model_display_name $display_name

echo "evaluating review ..."
python test.py --model_type $1 --resume --out_name review_str_zh$display_name.json --out_dir work_dirs/$display_name/ --dataset_path data/review_str_v1.json --eval review --prompt_type str --hf_path $hf_path --model_display_name $display_name

echo "evaluating plan ..."
python tools/eval/test.py --model_type $1 --resume --out_name plan_json_zh_$1.json --out_dir data/work_dirs/ --dataset_path data/plan_json_v1_zh.json --eval plan --prompt_type json --test_num $2
python tools/eval/test.py --model_type $1 --resume --out_name plan_str_zh_$1.json --out_dir data/work_dirs/ --dataset_path data/plan_str_v1_zh.json --eval plan --prompt_type str --test_num $2

echo "evaluating reason ..."
python tools/eval/test.py --model_type $1 --resume --out_name reason_retrieve_understand_json_zh_$1.json --out_dir data/work_dirs/ --dataset_path data/reason_retrieve_understand_json_v1_zh.json --eval reason --prompt_type json --test_num $2
python tools/eval/test.py --model_type $1 --resume --out_name reason_str_zh_$1.json --out_dir data/work_dirs/ --dataset_path data/reason_str_v1_zh.json --eval reason --prompt_type str --test_num $2

echo "evaluating retrieve ..."
python tools/eval/test.py --model_type $1 --resume --out_name reason_retrieve_understand_json_zh_$1.json --out_dir data/work_dirs/ --dataset_path data/reason_retrieve_understand_json_v1_zh.json --eval retrieve --prompt_type json --test_num $2
python tools/eval/test.py --model_type $1 --resume --out_name retrieve_str_zh_$1.json --out_dir data/work_dirs/ --dataset_path data/retrieve_str_v1_zh.json --eval retrieve --prompt_type str --test_num $2

echo "evaluating understand ..."
python tools/eval/test.py --model_type $1 --resume --out_name reason_retrieve_understand_json_zh_$1.json --out_dir data/work_dirs/ --dataset_path data/reason_retrieve_understand_json_v1_zh.json --eval understand --prompt_type json --test_num $2
python tools/eval/test.py --model_type $1 --resume --out_name understand_str_zh_$1.json --out_dir data/work_dirs/ --dataset_path data/understand_str_v1_zh.json --eval understand --prompt_type str --test_num $2