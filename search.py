from rank_bm25 import BM25Okapi
from gensim import similarities
from gensim.models import Doc2Vec
from gensim.corpora import Dictionary
import gensim.utils
from collections import namedtuple
from scipy import spatial
import json
import argparse
from tqdm import tqdm


def read_data(source):
    with open(source) as f:
        db = list(json.load(f))
    return db


def search_bm25(data):
    with open("nbc_documents.json", "r") as infile:
        corpus = list(json.load(infile))
    corpus = [text["document"] for text in corpus]
    texts = [gensim.utils.simple_preprocess(text) for text in corpus]
    bm25 = BM25Okapi(texts)

    correct = 0
    top3 = 0
    top5 = 0
    top10 = 0
    for entry in tqdm(data):
        query = gensim.utils.simple_preprocess(entry["question"])
        indices = [texts.index(doc) for doc in bm25.get_top_n(query, texts, n=10)]
        if corpus[indices[0]] == entry["context"]:
            correct += 1
            top3 += 1
            top5 += 1
            top10 += 1
        elif corpus[indices[1]] == entry["context"] or corpus[indices[2]] == entry["context"]:
            top3 += 1
            top5 += 1
            top10 += 1
        elif corpus[indices[3]] == entry["context"] or corpus[indices[4]] == entry["context"]:
            top5 += 1
            top10 += 1
        elif corpus[indices[5]] == entry["context"] or corpus[indices[6]] == entry["context"] or corpus[indices[7]] == entry["context"] or corpus[indices[8]] == entry["context"] or corpus[indices[9]] == entry["context"]:
            top10 += 1

    print(f"Lexical Search Accuracy (Top 1): {correct / len(data) * 100}")
    print(f"Lexical Search Accuracy (Top 3): {top3 / len(data) * 100}")
    print(f"Lexical Search Accuracy (Top 5): {top5 / len(data) * 100}")
    print(f"Lexical Search Accuracy (Top 10): {top10 / len(data) * 100}")
    return


def search_doc2vec(data):
    with open("nbc_documents.json", "r") as infile:
        corpus = list(json.load(infile))
    model = gensim.models.doc2vec.Doc2Vec.load("embeddings")
    corpus = [{"document": entry["document"], "vector": model.infer_vector(gensim.utils.simple_preprocess(entry["document"]))} for entry in corpus]

    correct = 0
    top3 = 0
    top5 = 0
    top10 = 0

    for entry in tqdm(data):
        query = gensim.utils.simple_preprocess(entry["question"])
        inferred_vector = model.infer_vector(query)
        sims_docs = []
        for value in corpus:
            sims_docs.append((1 - spatial.distance.cosine(value["vector"], inferred_vector), value["document"]))
            sims_docs.sort(reverse=True, key=lambda x: x[0])
        if sims_docs[0][1] == entry["context"]:
            correct += 1
            top3 += 1
            top5 += 1
            top10 += 1
        elif sims_docs[1][1] == entry["context"] or sims_docs[2][1] == entry["context"]:
            top3 += 1
            top5 += 1
            top10 += 1
        elif sims_docs[3][1] == entry["context"] or sims_docs[4][1] == entry["context"]:
            top5 += 1
            top10 += 1
        elif sims_docs[5][1] == entry["context"] or sims_docs[6][1] == entry["context"] or sims_docs[7][1] == entry["context"] or sims_docs[8][1] == entry["context"] or sims_docs[9][1] == entry["context"]:
            top10 += 1

    print(f"Semantic Search Accuracy: {correct / len(data) * 100}")
    print(f"Semantic Search Accuracy (Top 3): {top3 / len(data) * 100}")
    print(f"Semantic Search Accuracy (Top 5): {top5 / len(data) * 100}")
    print(f"Semantic Search Accuracy (Top 10): {top10 / len(data) * 100}")
    return


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth', '--t', nargs=1, required=True, type=str, help='file containing labeled answers')
    options = parser.parse_args()
    return options


def main():
    options = parse_options()
    data = read_data(options.ground_truth[0])
    search_bm25(data)
    search_doc2vec(data)
    return


if __name__ == "__main__":
    main()
