import numpy as np
import pandas as pd
import json
import argparse
import gensim
from gensim.models import Doc2Vec
from gensim import similarities
from gensim.corpora import Dictionary
import gensim.utils
from collections import namedtuple
import collections
from scipy import spatial
from rank_bm25 import BM25Okapi
import time
from tqdm import tqdm
from matplotlib import pyplot as plt


def read_data(source):
    with open(source) as f:
        db = list(json.load(f))
    return db


def compute_f1(a_gold, a_pred):
    gold_toks = gensim.utils.simple_preprocess(a_gold)
    pred_toks = gensim.utils.simple_preprocess(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def measure_results(ground_truth_answers, generated_answers, model):
    model = gensim.models.doc2vec.Doc2Vec.load(model)

    semantic_similarity = 0
    lexical_similarity = 0

    for k in tqdm(range(10)):
        for i in tqdm(range(len(ground_truth_answers))):
            semantic_similarities = []
            lexical_similarities = []
            for j in range(4):
                semantic_similarities.append(1 - spatial.distance.cosine(model.infer_vector(gensim.utils.simple_preprocess(ground_truth_answers[i]["choices"][j])), model.infer_vector(gensim.utils.simple_preprocess(generated_answers[i]["answer"]))))
                lexical_similarities.append(compute_f1(ground_truth_answers[i]["choices"][j], generated_answers[i]["answer"]))
            if max(semantic_similarities) == semantic_similarities[0]:
                semantic_similarity += 1
            if max(lexical_similarities) == lexical_similarities[0]:
                lexical_similarity += 1
    
    print(f"Average Semantic Similarity: {semantic_similarity / len(ground_truth_answers) / 10}")
    print(f"Average Lexical Similarity: {lexical_similarity / len(ground_truth_answers) / 10}")
    return semantic_similarity, lexical_similarity


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', '--t', nargs=1, required=True, type=str, help='file containing labeled answers')
    parser.add_argument('--generated', '--g', nargs=1, required=True, type=str, help='file containing generated answers')
    parser.add_argument('--embeddings', '--e', nargs=1, required=True, type=str, help='file to load embeddings model from')
    options = parser.parse_args()
    return options


def main():
    options = parse_options()
    measure_results(read_data(options.ground_truth[0]), read_data(options.generated[0]), options.embeddings[0])
    return


if __name__ == "__main__":
    main()
