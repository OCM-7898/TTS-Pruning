from collections import Counter
import os
import re
from typing import Dict, List, Optional
from math import comb

import datasets

from lm_eval.utils import eval_logger

if os.getenv("PROMPTSTEP") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTSTEP") + ' steps.'
elif os.getenv("PROMPTTOKEN") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTTOKEN") + ' tokens.'
elif os.getenv("PROMPTLONG") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double-check your work until you are absolutely sure that you have the correct answer.'
elif os.getenv("PROMPTSHORT") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a short amount of thinking. Do not spend excessive time double-checking your work.'
else:
    QUERY_TEMPLATE = '{Question}'

print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"

# Import the LLM sampler from AIME (if using LLM-as-a-judge)
EXTRACTION_TEMPLATE_IDX = r"""
Look at the following attempt by a student and extract the student's answer. If it is equivalent (ignoring trivial simplifications) to any of the provided options, return the index of that option starting from 1. Else, return -1.

Examples:

    Options: ['72,000']
    Attempt: 72000 \text{ cents}.

1
(always give benefit of the doubt to units and ignore formatting which makes the 1st option match)

    Options: ['27']
    Attempt: The answer is 27 miles.

1

---

YOUR TASK

Respond with only the index of the matching option starting from 1 or -1 if there is absolutely no reasonable match. Do not include a rationale.

    Options: %(expression1)s
    Attempt: %(expression2)s
""".strip()

def extract_answer_idx(sampler, options: List[str], attempt: str):
    prompt = EXTRACTION_TEMPLATE_IDX % {"expression1": options, "expression2": attempt}
    response = sampler([dict(content=prompt, role="user")])
    return response

import time
from typing import Any

import openai
from openai import OpenAI

class ChatCompletionSampler:
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI(api_key=os.environ.get(self.api_key_name))
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _pack_message(self, role: str, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1

def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc.get("question"))

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:

        out_doc = {
            "problem": doc.get("question"),  
            "solution": None, 
            "answer": doc.get("answer"),
        }
        return out_doc
    return dataset.map(_process_doc)

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    metrics = {"exact_match": None, "extracted_answers": []}
    
    # Multiple results -> we are measuring cov/maj/pass etc
    if isinstance(results[0], list):
        results = results[0]
        n_res = len(results)
        n_res_list = [2**i for i in range(1, int(n_res.bit_length()))]
        metrics = {
            **metrics,
            "exact_matches": [],
            **{f"cov@{n}": -1 for n in n_res_list},
            **{f"maj@{n}": -1 for n in n_res_list},
            **{f"pass@{n}": -1.0 for n in n_res_list},
        }

    # LLM-as-a-judge setup (optional)
    if os.getenv("PROCESSOR", "") == "gpt-4o-mini":
        sampler = ChatCompletionSampler(model="gpt-4o-mini")
    else:
        print(f"No PROCESSOR set; using exact string matching. Set 'PROCESSOR=gpt-4o-mini' and 'OPENAI_API_KEY=YOUR_KEY' for LLM-based matching.")
        sampler = None

   
    if isinstance(doc["answer"], str) and doc["answer"].isdigit():
        gt = str(int(doc["answer"]))
    else:
        gt = str(doc["answer"])
    
    split_tokens = ["<|im_start|>answer\n", "<|im_start|>"]

    for i, a in enumerate(results, start=1):
        # Handle special tokens
        if split_tokens[0] in a:
            a = a.split(split_tokens[0])[-1]
        elif split_tokens[1] in a:
            a = a.split(split_tokens[1])[-1]
            if "\n" in a:
                a = "\n".join(a.split("\n")[1:])

      
        if (box := last_boxed_only_string(a)) is not None:
            a = remove_boxed(box)
        
        elif (matches := re.findall(ANSWER_PATTERN, a, re.DOTALL)) != []:
            a = matches[-1]

        if (a.isdigit()) and (gt.isdigit()):
            a = str(int(a))  
        elif sampler is not None:
            options = [gt] + list(set(metrics["extracted_answers"]) - {gt})
            if len(options) > 7:
                print("Warning: Lots of options which may harm indexing performance:", options)
            options_str = "[" + ", ".join(["'" + str(o) + "'" for o in options]) + "]"
            idx = extract_answer_idx(sampler, options_str, a)
            if idx != "-1":
                if idx.isdigit():
                    idx = int(idx) - 1
                    if len(options) > idx >= 0:
                        a = options[idx]
                    else:
                        print("Warning: Index out of bounds; leaving answer unchanged\n", a, "\noptions", options_str, "\ndoc['answer']", gt, "\nidx", idx)
                else:
                    print("Warning: Processing did not produce integer index\na", a, "\noptions", options_str, "\ndoc['answer']", gt, "\nidx", idx)

        metrics["extracted_answers"].append(a)
        a = int(a == gt)
        if not(a):  
            print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + gt)
        
        if i == 1:
            metrics["exact_match"] = a
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(a)
        elif i > 1:
            metrics["exact_matches"].append(a)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(gt == Counter(metrics["extracted_answers"]).most_common(1)[0][0])
            c = sum(metrics["exact_matches"])
            for k in n_res_list:
                if c > 0:
                    metrics[f"pass@{k}"] = 1.0 - (comb(n_res - c, k) / comb(n_res, k))
                else:
                    metrics[f"pass@{k}"] = 0.0

    return metrics

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"
    return s[len(left) : -1]