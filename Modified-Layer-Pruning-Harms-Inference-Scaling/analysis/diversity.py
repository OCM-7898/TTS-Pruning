import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import argparse
import json
import os

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt_tab')

def calculate_self_bleu(jsonl_path):
    """
    Calculates the Self-BLEU score for a given jsonl path to measure repetition.

    A higher score indicates more repetition.
    A lower score indicates more diversity.
    """
    self_bleu_scores = []

    with open(jsonl_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line)
        model_response = data.get("resps", [[""]])[0][0]
        sentences = sent_tokenize(model_response.strip())

        if len(sentences) <= 1:
            self_bleu_scores.append(0.0) # Not enough sentences to compare

        total_bleu_score = 0.0
        smoothie = SmoothingFunction().method4

        for i, hypothesis_str in enumerate(sentences):
            references_str = sentences[:i] + sentences[i+1:]
            
            hypothesis = word_tokenize(hypothesis_str)
            references = [word_tokenize(sent) for sent in references_str]

            score = sentence_bleu(references, hypothesis, smoothing_function=smoothie)
            total_bleu_score += score

        self_bleu_scores.append(total_bleu_score / len(sentences))
    return np.average(self_bleu_scores)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--jsonl_path', type=str, help='Path to your jsonl file to analyse')

    args = parser.parse_args()

    if os.path.isfile(args.jsonl_path):
        print("Using file:", args.jsonl_path.split('/')[-1])
        self_bleu_score = calculate_self_bleu(args.jsonl_path)
        print("Self-BLEU Score:", self_bleu_score)
    else:
        print("No .jsonl file found in the folder:", {args.path})

if __name__ == '__main__':
    main()
