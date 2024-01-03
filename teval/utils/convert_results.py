import mmengine
import os
import argparse
import numpy as np
np.set_printoptions(precision=1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='work_dirs/nanbeige-agent/nanbeige-agent_-1.json')
    args = parser.parse_args()
    return args

def convert_results(result_path):
    result = mmengine.load(result_path)
    instruct_list = [(result['instruct_json']['json_format_metric'] + result['instruct_json']['json_args_em_metric']) / 2, (result['instruct_json']['string_format_metric'] + result['instruct_json']['string_args_em_metric']) / 2]
    plan_list = [result['plan_str']['f1_score'], result['plan_json']['f1_score']]
    reason_list = [result['reason_str']['thought'], result['reason_json']['thought']]
    retrieve_list = [result['retrieve_str']['name'], result['retrieve_json']['name']]
    understand_list = [result['understand_str']['args'], result['understand_json']['args']]
    review_list = [result['review_str']['review_quality'], result['review_str']['review_quality']]

    final_score = [np.mean(instruct_list), np.mean(plan_list), np.mean(reason_list), np.mean(retrieve_list), np.mean(understand_list), np.mean(review_list)]
    overall = np.mean(final_score)
    final_score.insert(0, overall)
    print(np.array(final_score) * 100)

if __name__ == '__main__':
    args = parse_args()
    convert_results(args.result_path)