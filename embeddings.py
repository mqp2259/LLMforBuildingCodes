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
import time
from tqdm import tqdm


def read_data(source):
    with open(source) as f:
        db = list(json.load(f))
    return db


def doc2vec_embed(source, destination):
    corpus = read_data(source)
    corpus = [gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(entry["content"]), [i]) for i, entry in enumerate(corpus)]
    print(len(corpus))

    model = gensim.models.Doc2Vec(
        dm=0,
        vector_size=200,
        min_count=5,
        epochs=25,
        window=10,
        max_final_vocab=1000000,
    )
    model.build_vocab(corpus)
    x = time.time()
    model.train(
        corpus,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )
    print(time.time() - x)

    ranks = []
    for doc_id in tqdm(range(len(corpus))):
        inferred_vector = model.infer_vector(corpus[doc_id].words)
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
    counter = collections.Counter(ranks)
    print(counter)

    model.save(destination)
    return


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '--s', nargs=1, required=True, type=str, help='file containing context documents to be embedded or model to run')
    parser.add_argument('--embeddings', '--e', nargs=1, required=True, type=str, help='file to store model in or load model from')
    options = parser.parse_args()
    return options


def main():
    options = parse_options()
    doc2vec_embed(options.source[0], options.embeddings[0])
    return


if __name__ == "__main__":
    main()
