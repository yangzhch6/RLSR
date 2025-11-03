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
import json
import pandas as pd
    
import datasets

SYSTEM_PROMPT = "Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: \"<think>\n {thoughts} </think>\n\". Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. After \"</think>\n,\" in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. If applicable, include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions."


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

    
def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/val")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "MATH_Olympiad_Minerva_AMC_AIME2425"

    dataset = load_json("/mnt/weka/home/yongxin.wang/workspace/lark/Datasets/valid_all.json")
    """
    {
        "problem": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$",
        "think_solution": null,
        "solution": null,
        "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
        "data_source": "math"
    },
    """
    # Convert train_dataset
    train_dataset = []
    for line in dataset:
        train_dataset.append({
            "data_source": line["data_source"],
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": line["problem"],
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": line["answer"]},
            "extra_info": {
                "split": "test",
                "index": str(len(train_dataset)),
                "answer": line["answer"],
                "question": line["problem"],
            },
        })

    # convert train_dataset to parquet file
    # 将train_dataset转换为DataFrame
    df = pd.DataFrame(train_dataset)
    
    # 确保目录存在
    output_dir = os.path.expanduser(args.local_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为parquet文件
    output_file = os.path.join(output_dir, "valid_all.parquet")
    df.to_parquet(output_file, index=False)
    print(f"数据集已成功保存到: {output_file}")
    print(f"总共处理了 {len(train_dataset)} 条数据")