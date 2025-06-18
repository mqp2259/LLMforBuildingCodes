from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
import os
import argparse
import time
import json


""" load data that will be used to make questions """
def load_data(name):
    with open(name, "r") as file:
        data = list(json.load(file))
    return data


""" save text generated to a json file """
def save_results(name, results):
    with open(name, "w") as file:
        file.write(results)
    return


""" iterator to speed up huggingface pipeline """
def dataset_maker(data, template):
    for document in data:
        if (document["page"] >= 26 and document["page"] <= 40) or (document["page"] >= 113 and document["page"] <= 124) or (document["page"] >= 145 and document["page"] <= 156) or (document["page"] >= 170 and document["page"] <= 176) or (document["page"] >= 184 and document["page"] <= 191) or (document["page"] >= 228 and document["page"] <= 242) or (document["page"] >= 244 and document["page"] <= 257) or (document["page"] >= 259 and document["page"] <= 267) or (document["page"] >= 269 and document["page"] <= 279) or (document["page"] >= 285 and document["page"] <= 297) or (document["page"] >= 317 and document["page"] <= 328) or (document["page"] >= 504 and document["page"] <= 518) or (document["page"] >= 560 and document["page"] <= 571) or (document["page"] >= 600 and document["page"] <= 614) or (document["page"] >= 632 and document["page"] <= 635) or (document["page"] >= 740 and document["page"] <= 745) or (document["page"] >= 847 and document["page"] <= 855) or (document["page"] >= 868 and document["page"] <= 876) or (document["page"] >= 888 and document["page"] <= 899) or (document["page"] >= 902 and document["page"] <= 912) or (document["page"] >= 918 and document["page"] <= 928) or (document["page"] >= 944 and document["page"] <= 949) or (document["page"] >= 980 and document["page"] <= 984) or (document["page"] >= 990 and document["page"] <= 993) or (document["page"] >= 996 and document["page"] <= 1003):
            yield template.replace("{}", document["document"])
        else:
            continue


""" utilize new llama model to generate data """
def llama_generate(name, data):
    assert name in ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-hf", "meta-llama/Llama-2-70b-chat-hf"]    # only one of small models
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",    # finds GPU
    )

    # best template found so far, ensures proper format and acceptable quality of outputs
    template = "Generate exactly one open-ended question with one correct and three plausible incorrect answers based only on the content in the following document.\n\n\"{}\"\n\nAll answers must be written in full sentences and they must be of similar length. Respond in the following format:\n\nQuestion: <question>\n\nA: <correct answer to question>\nB: <incorrect answer 1 to question>\nC: <incorrect answer 2 to question>\nD: <incorrect answer 3 to question>\n\nAnswer: <letter corresponding to correct answer>\nExplanation: <justification of answer>\n\nQuestion:"
    results = []
    c_time = time.time()

    # generate in a loop using baseline settings
    with torch.no_grad():    # no gradients are saved automatically, ensures we don't hit CUDA OOM
        for sequences in generation_pipe(dataset_maker(data, template), max_length=1280, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True, top_k=10, temperature=0.4, top_p=0.9):
            id_no = len(results)
            text = sequences[0]["generated_text"]
            text_length = len(tokenizer.encode(text))
            generate_time = time.time() - c_time
            tokens_per_second = text_length / generate_time
            # save and print all data about the generation in json format
            results.append(json.dumps(
                {
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


""" parse command line invocation """
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', '--f', nargs=1, type=str, required=True, choices=['gpt', 'falcon', 'llama2'], help='model family')
    parser.add_argument('--model', '--m', nargs=1, type=str, required=True, choices=["gpt-4", "gpt-3.5-turbo", "tiiuae/falcon-40b-instruct", "tiiuae/falcon-40b", "tiiuae/falcon-7b-instruct", "tiiuae/falcon-7b", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-hf", "meta-llama/Llama-2-70b-chat-hf"], help='specific model')
    parser.add_argument('--data', '--d', nargs=1, type=str, required=True, help='documents to generate on')
    parser.add_argument('--results', '--r', nargs=1, type=str, required=True, help='where to save results')
    options = parser.parse_args()
    return options


""" generate data """
def main():
    options = parse_options()
    data = load_data(options.data[0])
    results = llama_generate(options.model[0], data)
    save_results(options.results[0], results)
    return


if __name__ == "__main__":
    main()
