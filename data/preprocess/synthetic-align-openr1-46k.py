# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import pandas as pd
import datasets

SYS_PROMPT = """You are a math AI assistant. You need to solve the given math problem. For calculation problem, show your work clearly and put your final answer within \\boxed{}. For proof problem, provide a rigorous logical derivation. Ensure your solution is clearly stated. (You can only use natural language, not formal language.)

You will be provided with a Question and its Ground Truth Solution. In order to guarantee accuracy, you should consult the ground truth to inform your thought process and your own solution. However, it is imperative that you **NEVER** reference or suggest the existence of a "ground truth" in your thought process or final solution. **You must solve/prove the problem as if you are reasoning from scratch, solely relying on your own abilities.**
"""

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def contain_think(response):
    if response.count("<think>") == 1 and response.count("</think>") == 1:
        return True
    return False

def filter(line):
    response = line["target"][0]["content"]
    if contain_think(response):
        return True
    return False

def extract_question(line):
    return line["prompt"][1]["content"]

def extract_solution(line):
    return line["target"][0]["content"].split("</think>")[-1].strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/synthetic_align/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "Elliott/Openr1-Math-46k-8192"

    dataset = datasets.load_dataset(data_source, split="train")
    print(dataset[0])

    processed_data = []

    for line in dataset:
        if not filter(line):
            continue
        
        question = extract_question(line)
        solution = extract_solution(line)

        processed_data.append({
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": SYS_PROMPT,
                },
                {
                    "role": "user",
                    "content": "## Question:\n{}\n\n\n## Ground Truth Solution:\n{}".format(question, solution)
                }
            ],
            "response": line["target"][0]["content"],
            "ability": "math",
            "extra_info": {
                "split": "train",
                "index": str(len(processed_data)),
            },

        })

    print(processed_data[0])
    
    df = pd.DataFrame(processed_data)
    # 确保目录存在
    output_dir = os.path.expanduser(args.local_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 保存为parquet文件
    output_file = os.path.join(output_dir, "openr1-46k.parquet")
    df.to_parquet(output_file, index=False)
    print(f"数据集已成功保存到: {output_file}")
    print(f"总共处理了 {len(processed_data)} 条数据")


    # 选取128条数据，保存为openr1-46k-val.parquet
    df = pd.DataFrame(processed_data[:128])
    # 确保目录存在
    output_dir = os.path.expanduser(args.local_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 保存为parquet文件
    output_file = os.path.join(output_dir, "openr1-46k-val.parquet")
    df.to_parquet(output_file, index=False)
