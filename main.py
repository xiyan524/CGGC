import json
import os
import argparse
import random
import numpy as np

import fire
#from llama import Llama
from typing import List

import torch
#print("Cuda is available:", torch.cuda.is_available())
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def data_load(data_path):
    with open(data_path, "r") as f:
        data_lines = f.readlines()
        all_data = [json.loads(data) for data in data_lines]

    return all_data

def write_to_file(results, output_path):
    with open(output_path, "a") as f:
        for result in results:
            f.write(json.dumps(result))
            f.write("\n")

def inference_batch(prompts, model, tokenizer, max_seq_gen, batch_data, output_path):
    """do in-context learning in a batch"""
    try:
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=max_seq_gen)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except Exception as e:
        print(e)
        response = [""] * len(prompts)

    results = []
    for index,item in enumerate(response):
        data = batch_data[index]
        data['preds'] = item
        results.append(data)
    write_to_file(results, output_path)

def do_inference(all_data, max_batch_size, max_seq_gen, tokenizer, model, output_path, icl_num, prompt_str, demonstration_type, model_name, prompt_type):
    batch_data = []
    batch_prompt = []
    for index, data in enumerate(all_data):

        # filter data which has small approriate demonstrations and construct prompts
        if len(data[demonstration_type]) < icl_num:
            continue
        else:
            demonstrations = data[demonstration_type]
            batch_data.append(data)

            def construct_cps_graph_prompt(concepts, graph):
                random.shuffle(graph)
                cps_str = "### concepts: " + ", ".join(concepts) + "\n "
                graph_str = "### graph: "
                for edge in graph:
                    if edge[3] == "left":
                        graph_str += edge[0].replace("/c/en/", "") + " - " + edge[2].replace("/r/", "") + " - " +  edge[1].replace("/c/en/", "") + ", "
                    else:
                        graph_str += edge[1].replace("/c/en/", "") + " - " + edge[2].replace("/r/", "") + " - " +  edge[0].replace("/c/en/", "") + ", "
                graph_str += "\n "
                return cps_str + graph_str

            tmp_prompt = prompt_str
            if model_name == "mistral-instruct":
                tmp_prompt = '<s>[INST] ' + tmp_prompt  # for using instruction
            for idx in range(icl_num):
                try:
                    example = demonstrations[idx]
                    tmp_prompt += construct_cps_graph_prompt(example['concepts'], example['pruned_graph']) + "### sentence: " + example['target'] + "; "
                except:
                    example = demonstrations[idx]
                    print(example)

            tmp_prompt += construct_cps_graph_prompt(data['concepts'], data['pruned_graph']) + "### sentence:"
            batch_prompt.append(tmp_prompt)


        # do inference in a batch for efficiency
        if len(batch_data) == max_batch_size:
            inference_batch(batch_prompt, model, tokenizer, max_seq_gen, batch_data, output_path)
            batch_data = []
            batch_prompt = []

    # last batch
    if len(batch_data) > 0:
        response = inference_batch(batch_prompt, model, tokenizer, max_seq_gen, batch_data, output_path)


def main(
    data_path: str,
    max_batch_size: int,
    max_seq_gen: int,
    output_path: str,
    icl_num: int,
    model_name: str,
    prompt_str: str,
    demonstration_type: str,
    prompt_type: str,
    random_seed: int,
):

    set_seed(random_seed)
    all_data = data_load(data_path)
    print("data size:", len(all_data))

    MODELS = {
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'llama2-7b-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'mistral': 'mistralai/Mistral-7B-v0.1',
    'mistral-instruct': 'mistralai/Mistral-7B-Instruct-v0.1',
    'falcon': 'tiiuae/falcon-7b',
    'falcon-instruct': 'tiiuae/falcon-7b-instruct',
    'gpt-j': 'EleutherAI/gpt-j-6B',
    }

    dtype = torch.float32 if 'llama2-7b' in model_name else torch.float16 
    with torch.no_grad(): 
        model = AutoModelForCausalLM.from_pretrained(MODELS[model_name], torch_dtype=dtype, device_map="auto", token=True)
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name], use_fast=False, padding_side='left')
    if "falcon" in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    do_inference(all_data, max_batch_size, max_seq_gen, tokenizer, model, 
                output_path, icl_num, prompt_str, demonstration_type, model_name, prompt_type)


if __name__ == "__main__":
    fire.Fire(main)

