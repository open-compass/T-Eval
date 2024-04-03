import argparse
import os
import random
import shutil

import mmengine
import teval.evaluators as evaluator_factory
from lagent.llms.huggingface import HFTransformerCasualLM, HFTransformerChat
from lagent.llms.openai import GPTAPI
from lagent.llms.vllm_wrapper import VllmModel
from teval.utils.meta_template import meta_template_dict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/instruct_v1.json')
    parser.add_argument('--model_type', type=str, choices=['api', 'hf', 'vllm'], default='hf')
    # hf means huggingface, if you want to use huggingface model, you should specify the path of the model
    parser.add_argument('--model_display_name', type=str, default="")
    # if not set, it will be the same as the model type, only inference the output_name of the result
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--out_name', type=str, default='tmp.json')
    parser.add_argument('--out_dir', type=str, default="work_dirs/")
    # [api model name: 'gpt-3.5-turbo-16k', 'gpt-4-1106-preview', 'claude-2.1', 'chat-bison-001']
    parser.add_argument('--model_path', type=str, help="path to huggingface model / api model name")
    parser.add_argument('--eval', type=str, choices=['instruct', 'reason', 'plan', 'retrieve', 'review', 'understand', 'rru'])
    parser.add_argument('--test_num', type=int, default=-1, help='number of samples to test, -1 means all')
    parser.add_argument('--prompt_type', type=str, default='json', choices=['json', 'str'])
    parser.add_argument('--meta_template', type=str, default='qwen')
    parser.add_argument('--batch_size', type=int, default=1)
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

def split_special_tokens(text):
    text = text.split('<eoa>')[0]
    text = text.split('<TOKENS_UNUSED_1>')[0]
    text = text.split('<|im_end|>')[0]
    text = text.split('\nuser')[0]
    text = text.split('\nassistant')[0]
    text = text.split('\nUSER')[0]
    text = text.split('[INST]')[0]
    text = text.split('<|user|>')[0]
    text = text.strip()
    if text.startswith('```json'):
        text = text[len('```json'):]
    text = text.strip('`').strip()
    return text

def infer(dataset, llm, out_dir, tmp_folder_name='tmp', test_num = 1, batch_size=1):
    random_list = list(dataset.keys())[:test_num]
    batch_infer_list = []; batch_infer_ids = []
    for idx in tqdm(random_list):
        prompt = dataset[idx]['origin_prompt']
        batch_infer_list.append(prompt)
        batch_infer_ids.append(idx)
        # batch inference
        if len(batch_infer_ids) == batch_size or idx == len(random_list) - 1:
            predictions = llm.chat(batch_infer_list, do_sample=False)
            for ptr, prediction in enumerate(predictions):
                if not isinstance(prediction, str):
                    print("Warning: the output of llm is not a string, force to convert it into str")
                    prediction = str(prediction)
                prediction = split_special_tokens(prediction)
                data_ptr = batch_infer_ids[ptr]
                dataset[data_ptr]['prediction'] = prediction
                mmengine.dump(dataset[data_ptr], os.path.join(out_dir, tmp_folder_name, f'{data_ptr}.json'))
            batch_infer_ids = []; batch_infer_list = []

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
    dataset, tested_num, total_num = load_dataset(args.dataset_path, args.out_dir, args.resume, tmp_folder_name=tmp_folder_name)
    if args.test_num == -1:
        test_num = max(total_num - tested_num, 0)
    else:
        test_num = max(min(args.test_num - tested_num, total_num - tested_num), 0)
    output_file_path = os.path.join(args.out_dir, args.out_name)
    if test_num != 0:
        if args.model_type == 'api':
            # if you want to use GPT, please refer to lagent for how to pass your key to GPTAPI class
            llm = GPTAPI(args.model_path)
        # elif args.model_type.startswith('claude'):
        #     llm = ClaudeAPI(args.model_type)
        elif args.model_type == 'hf':
            meta_template = meta_template_dict.get(args.meta_template)
            if "chatglm" in args.model_display_name:
                llm = HFTransformerChat(path=args.model_path, meta_template=meta_template)
            else:
                llm = HFTransformerCasualLM(path=args.model_path, meta_template=meta_template, max_new_tokens=512)
        elif args.model_type == "vllm":
            llm = VllmModel(path=args.model_path, meta_template=None, max_new_tokens=512, top_k=1)
        else:
            raise ValueError
        print(f"Tested {tested_num} samples, left {test_num} samples, total {total_num} samples")
        prediction = infer(dataset, llm, args.out_dir, tmp_folder_name=tmp_folder_name, test_num=test_num, batch_size=args.batch_size)
        # dump prediction to out_dir
        mmengine.dump(prediction, os.path.join(args.out_dir, args.out_name))

    if args.eval:
        if args.model_display_name == "":
            model_display_name = args.model_type
        else:
            model_display_name = args.model_display_name
        os.makedirs(args.out_dir, exist_ok=True)
        eval_mapping = dict(
            instruct="InstructEvaluator",
            plan="PlanningEvaluator",
            review="ReviewEvaluator",
            reason="ReasonRetrieveUnderstandEvaluator",
            retrieve="ReasonRetrieveUnderstandEvaluator",
            understand="ReasonRetrieveUnderstandEvaluator",
            rru="ReasonRetrieveUnderstandEvaluator"
        )
        if "_zh" in args.dataset_path:
            bert_score_model = "thenlper/gte-large-zh"
            json_path = os.path.join(args.out_dir, model_display_name + '_' + str(args.test_num) + '_zh.json')
        else:
            bert_score_model = "all-mpnet-base-v2"
            json_path = os.path.join(args.out_dir, model_display_name + '_' + str(args.test_num) + '.json')
        evaluator_class = getattr(evaluator_factory, eval_mapping[args.eval])
        evaluator = evaluator_class(output_file_path, default_prompt_type=args.prompt_type, eval_type = args.eval, bert_score_model=bert_score_model)
        if os.path.exists(json_path):
            results = mmengine.load(json_path)
        else:
            results = dict()
        eval_results = evaluator.evaluate()
        print(eval_results)
        results[args.eval + '_' + args.prompt_type] = eval_results
        print(f"Writing Evaluation Results to {json_path}")
        mmengine.dump(results, json_path)
