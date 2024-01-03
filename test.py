import teval.evaluators as evaluator_factory
from teval.utils.meta_template import meta_template_dict
from lagent.llms.huggingface import HFTransformerCasualLM
from lagent.llms.openai import GPTAPI
import argparse
import mmengine
import os
from tqdm import tqdm
import shutil
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/instruct_v1.json')
    parser.add_argument('--model_type', type=str, choices=['gpt-3.5-turbo-16k', 'gpt-4-1106-preview', 'hf', 'claude-2.1', 'chat-bison-001'], default='gpt-3.5-turbo-16k')
    # hf means huggingface, if you want to use huggingface model, you should specify the path of the model
    parser.add_argument('--model_display_name', type=str, default="")
    # if not set, it will be the same as the model type, only inference the output_name of the result
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--out_name', type=str, default='tmp.json')
    parser.add_argument('--out_dir', type=str, default="work_dirs/")
    parser.add_argument('--hf_path', type=str, help="path to huggingface model")
    parser.add_argument('--eval', type=str, choices=['instruct', 'reason', 'plan', 'retrieve', 'review', 'understand'])
    parser.add_argument('--test_num', type=int, default=-1, help='number of samples to test, -1 means all')
    parser.add_argument('--prompt_type', type=str, default='json', choices=['json', 'str'])
    parser.add_argument('--meta_template', type=str, default='internlm')
    args = parser.parse_args()
    return args

def load_dataset(dataset_path, out_dir, is_resume=False, tmp_folder_name='tmp'):
    dataset = mmengine.load(dataset_path)
    total_num = len(dataset)
    # possible filter here
    tested_num = 0
    if is_resume:
        file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
        for filename in file_list:
            if filename.split('.')[0] in dataset:
                tested_num += 1
                file_id = filename.split('.')[0]
                dataset.pop(file_id)
            else:
                print(f"Warning: {filename} not in dataset, remove it from cache")
                os.remove(os.path.join(out_dir, tmp_folder_name, filename))

    return dataset, tested_num, total_num

def infer(dataset, llm, out_dir, tmp_folder_name='tmp', test_num = -1):
    random_list = list(dataset.keys())
    random.shuffle(random_list)
    for idx in tqdm(random_list):
        if test_num == 0:
            break
        test_num -= 1
        prompt = dataset[idx]['origin_prompt']
        prediction = llm.generate_from_template(prompt, 1024)
        dataset[idx]['prediction'] = prediction
        mmengine.dump(dataset[idx], os.path.join(out_dir, tmp_folder_name, f'{idx}.json'))
    # load results from cache
    results = dict()
    file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
    for filename in file_list:
        file_id = filename.split('.')[0]
        results[file_id] = mmengine.load(os.path.join(out_dir, tmp_folder_name, filename))
    return results
    
if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_folder_name = os.path.splitext(args.out_name)[0]
    os.makedirs(os.path.join(args.out_dir, tmp_folder_name), exist_ok=True)
    if args.model_type.startswith('gpt'):
        # if you want to use GPT, please refer to lagent for how to pass your key to GPTAPI class
        llm = GPTAPI(args.model_type)
    # elif args.model_type.startswith('claude'):
    #     llm = ClaudeAPI(args.model_type)
    elif args.model_type == 'hf':
        meta_template = meta_template_dict.get(args.meta_template)
        llm = HFTransformerCasualLM(args.hf_path, meta_template=meta_template)
    dataset, tested_num, total_num = load_dataset(args.dataset_path, args.out_dir, args.resume, tmp_folder_name=tmp_folder_name)
    if args.test_num == -1:
        test_num = max(total_num - tested_num, 0)
    else:
        test_num = max(min(args.test_num - tested_num, total_num - tested_num), 0)
    print(f"Tested {tested_num} samples, left {test_num} samples, total {total_num} samples")
    prediction = infer(dataset, llm, args.out_dir, tmp_folder_name=tmp_folder_name, test_num=test_num)
    # dump prediction to out_dir
    output_file_path = os.path.join(args.out_dir, args.out_name)
    mmengine.dump(prediction, os.path.join(args.out_dir, args.out_name))

    if args.eval:
        if args.model_display_name == "":
            model_display_name = args.model_type
        else:
            model_display_name = args.model_display_name
        print(model_display_name)
        os.makedirs(args.out_dir, exist_ok=True)
        json_path = os.path.join(args.out_dir, model_display_name + '_' + str(args.test_num) + '.json')
        if os.path.exists(json_path):
            results = mmengine.load(json_path)
        else:
            results = dict()
        eval_mapping = dict(
            instruct="InstructEvaluator",
            plan="PlanningEvaluator",
            review="ReviewEvaluator",
            reason="ReasonRetrieveUnderstandEvaluator",
            retrieve="ReasonRetrieveUnderstandEvaluator",
            understand="ReasonRetrieveUnderstandEvaluator"
        )
        evaluator_class = getattr(evaluator_factory, eval_mapping[args.eval])
        evaluator = evaluator_class(output_file_path, default_prompt_type=args.prompt_type, eval_type = args.eval)

        eval_results = evaluator.evaluate()
        print(eval_results)
        results[args.eval + '_' + args.prompt_type] = eval_results
        print(json_path)
        mmengine.dump(results, json_path)