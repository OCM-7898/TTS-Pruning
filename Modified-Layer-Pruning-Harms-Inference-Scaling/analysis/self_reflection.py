import argparse
import json
import openai
import os
import re
import time
import numpy as np
from tqdm import tqdm
from zhipuai import ZhipuAI  # pip install zhipuai

# client = ZhipuAI(api_key="YOUR_API_KEY")
# MODEL_NAME = "glm-4.5-flash"
# OpenAI API
client = openai.OpenAI(api_key="YOUR_API_KEY")
MODEL_NAME = "gpt-4o-mini"

# === Prompt templates with placeholder ===
def make_prompts(completion):
    return [
        f"""Here is a chain-of-reasoning that a language model generated while trying to solve a problem from the AIME24 dataset.

Chain-of-reasoning:
{completion}

Evaluate whether the chain-of-reasoning contains any **answer-verification steps**, where the model explicitly or implicitly checks whether a result matches the target answer.

Examples of **explicit verification** include:
- "This gives 1, which is not equal to 22"
- "Since 25 is not equal to 22..."

Examples of **implicit verification** include:
- "Too high!"
- "This works!"

If the reasoning contains large portions of repeated or identical text (e.g., the same sentences or phrases repeated multiple times), treat it as repetition noise and only count as once.

If you find any answer-verification steps, count the total number and provide the result between the tags <count> </count>. If none are found, return <count>0</count>.""",

        # 2. Backtracking
        f"""Here is a chain-of-reasoning that a language model generated while trying to solve a problem from the AIME24 dataset.

Chain-of-reasoning:
{completion}

Evaluate whether the chain-of-reasoning contains any **backtracking behavior**, where the model realizes that a current approach is not working and explicitly tries a different strategy or path.

In this context, backtracking includes discarding a partial solution and restarting from a previous step or exploring a new computation route not derived from the immediately prior result.

Example:
- The reasoning tries (a + b), then later tries (a × c) or (b - d) without using the previous intermediate result.

If the reasoning contains large portions of repeated or identical text, treat it as repetition noise and only count as once.

Count the number of distinct backtracking instances and provide the result between the tags <count> </count>. If none are found, return <count>0</count>.""",

        # 3. Subgoal Setting
        f"""Here is a chain-of-reasoning that a language model generated while trying to solve a problem from the AIME24 dataset.

Chain-of-reasoning:
{completion}

Evaluate whether the chain-of-reasoning contains any **explicit subgoal setting**, where the model breaks down the main problem into smaller intermediate goals or lemmas to help reach the final answer.

Examples of subgoal setting include:
- "First, I'll simplify the inner expression..."
- "Let’s prove that the denominator is positive before continuing..."
- "We can begin by factoring the polynomial..."
- "As a first step, assume x > 0 and consider that case..."

If the reasoning contains large portions of repeated or identical text, treat it as repetition noise and only count as once.

Count the number of distinct subgoals the model sets and provide the result between the tags <count> </count>. If none are found, return <count>0</count>.""",

]


# Extract <count> from model response
def extract_count(response_text):
    match = re.search(r"<count>(\d+)</count>", response_text)
    if not match:
        print(response_text, '\n')
    return int(match.group(1)) if match else 0

def analyze_reasoning(reasoning):
    prompts = make_prompts(reasoning)
    counts = []
    for prompt in prompts:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10000,
            )
            if MODEL_NAME.startswith("gpt-") or MODEL_NAME.startswith("glm-"):
                content = response.choices[0].message.content
            else:
                raise ValueError("Unsupported model type:", MODEL_NAME)
            count = extract_count(content)
            counts.append(count)

            # Avoid "Too Many Requests" error
            time.sleep(10)
        except Exception as e:
            print("Error during completion:", e)
            counts.append(0)
    return counts  # [verification, backtracking, subgoal]

def process_jsonl_and_analyze(jsonl_path, max_samples):
    results = []
    correct_results = []
    with open(jsonl_path, 'r') as f:
        lines = [json.loads(line) for line in f if line.strip()]

    if max_samples != -1:
        assert max_samples > 0, f"max_samples must be larger than 0, you give max_samples={max_samples}"
        sampled = lines[:max_samples]
    else:
        sampled = lines

    for entry in tqdm(sampled, desc="Analyzing"):
        doc_id = entry.get("doc_id", -1)
        resps = entry.get("filtered_resps") or entry.get("resps")
        if not resps or len(resps) == 0:
            continue
        reasoning_trace = resps[0][0].strip()
        counts = analyze_reasoning(reasoning_trace)
        results.append((doc_id, *counts))

    if results:
        data = np.array([r[1:] for r in results], dtype=float)
        averages = np.mean(data, axis=0)
        print("Average of each element (excluding doc_id):", averages)
    else:
        print("No results to compute averages.")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--jsonl_path', type=str, help='Path to your jsonl file to analyse')
    parser.add_argument('--max_samples', type=int, default=-1, help='Max samples')

    args = parser.parse_args()

    if os.path.isfile(args.jsonl_path):
        print("Using file", args.jsonl_path.split('/')[-1])
        process_jsonl_and_analyze(args.jsonl_path, args.max_samples)
    else:
        raise("No .jsonl file found in the folder:", {args.path})

if __name__ == "__main__":
    main()
