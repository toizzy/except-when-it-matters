"""
Run one iteration of the experiment, training on one language and testing on another.
"""
import argparse
import csv
import json
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
from transformers import AutoTokenizer, AutoModel

from utils import get_tokens_and_labels, get_tokens_and_labels_csv, \
    get_bert_tokens, shuffle_positions, save_sample, save_bert_outputs,
    save_just_position_word

base_path = "dataset_storing" 
def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ud-path', type=str,
        default=None)
    parser.add_argument('--csv-file', type=str, default=None, help="If data is in a csv file with subjects and objects marked")
    parser.add_argument('--bert-name', type=str, help="Like 'bert-base-uncased'")
    parser.add_argument('--shuffle-positions', action="store_true")
    parser.add_argument('--single-position', type=int, default=-1, 
        help="Make all positions this one index")
    parser.add_argument('--local-shuffle', type=int, default=-1)
    args = parser.parse_args()
    print("args:", args)
    make_dataset(args)

def make_dataset(args):
    if args.ud_path is not None:
        tb_name = os.path.split(args.ud_path)[1]
        tb_name = os.path.splitext(tb_name)[0]
        directory = os.path.join(base_path, f"{tb_name}_{args.bert_name}")
    if args.csv_file is not None:
        dataset_name = os.path.split(args.csv_file)[1]
        dataset_name = os.path.splitext(dataset_name)[0]
        directory = os.path.join(base_path, f"{dataset_name}_{args.bert_name}")
    if args.single_position >= 0:
        directory += f"_singlepos{args.single_position}"
    elif args.shuffle_positions:
        directory += f"_shuffled-pos"
    elif args.local_shuffle >= 0:
        directory += f"_localshuffle{args.local_shuffle}"
    os.mkdir(directory)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    model = AutoModel.from_pretrained(args.bert_name, output_hidden_states=True)
    model.eval()
    if args.ud_path is not None:
        labels = get_tokens_and_labels(args.ud_path)
    elif args.csv_file is not None:
        labels = get_tokens_and_labels_csv(args.csv_file)
    labels = shuffle_positions(labels, args.shuffle_positions, args.local_shuffle)
    json.dump(labels, open(os.path.join(directory, "labels.json"), "w"))
    save_sample(20, labels, directory)
    bert_info = {}
    bert_info["bert_tokens"], bert_info["bert_ids"], \
    bert_info["orig_to_bert_map"], bert_info["bert_to_orig_map"] =\
        get_bert_tokens(labels["token"], tokenizer)
    pickle.dump(bert_info, open(os.path.join(directory, "bert_info.pkl"), "wb"))
    bert_vectors_path = os.path.join(directory, "bert_vectors.hdf5")
    save_bert_outputs(directory, bert_info["bert_ids"], model, args.shuffle_positions, args.single_position)

if __name__ == "__main__":
    __main__()
