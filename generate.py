# import basic utilities
import os
import sys
import numpy as np
import pandas as pd
import json
import time
import argparse
# import llm tools needed for inferencing
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig,
)
from peft import (
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
)
# import tools for document searching
from rank_bm25 import BM25Okapi
from gensim import similarities
from gensim.models import Doc2Vec
from gensim.corpora import Dictionary
import gensim.utils
from collections import namedtuple


# save generated data
def save_results(name, results):
    with open(name, "w") as f:
        f.write(str(results))
    return


# implementation allowing for stop words to be respected
class StopOnTokens(StoppingCriteria):
    # store stop word information
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    # checks if a given sequence ends in one of the stop tokens
    def __call__(self, input_ids, scores, **kwargs):
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


# iterator to speed up huggingface pipeline
def dataset_maker(data, template, search):
    for document in data:
        question = document["question"]
        if search == "none":
            context = document["context"]
        elif search == "bm25":
            context = search_bm25(question)
        elif search == "doc2vec":
            context = search_doc2vec(question)
        prompt = f"Consider the following information: \"{context}\". {question}"
        yield template.replace("{}", prompt)


# search document corpus based on BM25 algorithm
def search_bm25(query):
    with open("nbc_documents.json", "r") as f:
        corpus = list(json.load(f))
    corpus = [text["document"] for text in corpus]
    texts = [gensim.utils.simple_preprocess(text) for text in corpus]
    bm25 = BM25Okapi(texts)
    query = gensim.utils.simple_preprocess(query)
    document = corpus[texts.index(bm25.get_top_n(query, texts, n=1)[0])]
    return document


# use embeddings-based search
def search_doc2vec(query):
    with open("nbc_documents.json", "r") as f:
        corpus = list(json.load(f))
    model = gensim.models.doc2vec.Doc2Vec.load("embeddings")
    inferred_vector = model.infer_vector(gensim.utils.simple_preprocess(query))
    document = corpus[model.dv.most_similar([inferred_vector], topn=1)[0][0]]
    return document


# stores an LLM and its functionality of any type
class LLM():
    # maps inputs into self contained variables
    def __init__(self, model_family, base_model, finetuned_model, quantization, search, testing_data):
        self.model_family = model_family
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.quantization = quantization
        self.search = search
        with open(testing_data, "r") as f:
            self.testing_data = list(json.load(f))

        self.stop_tokens = [["Human", ":"], ["AI", ":"]]
        self.stop_token_ids = None
        self.stopping_criteria = None
        
        self.template = None

        self.model = None
        self.tokenizer = None
        self.pipeline = None
        return

    # inferences the model in conversational format
    def chat(self):
        # load finetuned tokenizer and configure special tokens as stop words
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.stop_token_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in self.stop_tokens]
        self.stop_token_ids = [torch.LongTensor(x).to("cuda") for x in self.stop_token_ids]
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids)])

        # load base model and apply adapters from finetuning, set into evaluation/inference mode
        if self.model_family == 'falcon':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        elif self.model_family == 'llama2':
            self.model = LlamaForCausalLM.from_pretrained(
                self.base_model,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        if self.finetuned_model:
            self.model = PeftModel.from_pretrained(
                self.model,
                self.finetuned_model,
                torch_dtype=torch.float16,
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.eval()
        self.model = torch.compile(self.model)

        # configure model into a hugging face pipeline for easy interfacing
        if self.model_family == 'falcon':
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=128,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.stopping_criteria,
            )
        elif self.model_family == 'llama2':
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=self.stopping_criteria,
            )

        # configure prompt template to generate data
        self.template = "You are a helpful assistant who truthfully answers a human's questions based on provided context.\n\nHuman: {}\nAI: "
        results = []
        c_time = time.time()

        # generate in a loop using baseline settings
        with torch.no_grad():    # no gradients are saved automatically, so ensures we don't hit CUDA OOM
            for sequences in self.pipeline(dataset_maker(self.testing_data, self.template, self.search), max_new_tokens=256, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, stopping_criteria=self.stopping_criteria,):
                id_no = len(results)
                text = sequences[0]["generated_text"]
                text = text.removesuffix("Human:")
                text = text.removesuffix("AI:")
                while text[-1] == "\n":
                    text = text.removesuffix("\n")
                text_length = len(self.tokenizer.encode(text))
                generate_time = time.time() - c_time
                tokens_per_second = text_length / generate_time
                # save and print all data about the generation in json format
                results.append(json.dumps({
                        "id_no": id_no,
                        "text": text,
                        "text_length": text_length,
                        "generate_time": generate_time,
                        "tokens_per_second": tokens_per_second,
                    },
                    indent=4,
                ))
                print(results[-1] + ",")
                c_time = time.time()
        return results


# determine whetehr we should inference or train the llama model
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-family', '--f', nargs=1, type=str, required=True, help='model type')
    parser.add_argument('--base-model', '--b', nargs=1, type=str, required=True, help='pretrained model to finetune')
    parser.add_argument('--finetuned-model', '--o', nargs=1, type=str, required=True, help='location to save/read LLM in')
    parser.add_argument('--quantization', '--q', nargs=1, default=['8'], type=str, choices=['4', '8'], help='x-bit quantization of llm')
    parser.add_argument('--search', '--s', nargs=1, default=['none'], type=str, choices=['none', 'bm25', 'doc2vec'], help='search algorithm to use')
    parser.add_argument('--testing-data', '--d', nargs=1, type=str, required=True, help='search algorithm to use')
    parser.add_argument('--results', '--r', nargs=1, type=str, required=True, help='where to save results')
    options = parser.parse_args()
    return options


# execute the program based on user desire
def main():
    options = parse_options()
    llm = LLM(options.model_family[0], options.base_model[0], options.finetuned_model[0], options.quantization[0], options.search[0], options.testing_data[0])
    results = llm.chat()
    save_results(options.results[0], results)
    return


if __name__ == "__main__":
    main()
