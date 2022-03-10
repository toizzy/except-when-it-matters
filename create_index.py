
import argparse
from collections import defaultdict
import json
import os
import pickle
from random import shuffle
import sys
import torch

from utils import load_embeddings

base_path = "dataset_storing"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--roles', nargs='+', type=str, help="Roles, like A or S-passive. Should be capitalized correctly")
    parser.add_argument('--cases', nargs='+', type=str, help="Cases, like Erg or Nom. Should be capitalized correctly")
    parser.add_argument('--balance', action="store_true")
    parser.add_argument('--only-non-prototypical', action="store_true")
    parser.add_argument('--limit', type=int, default=-1)
    args = parser.parse_args()
    print(args)
    create_index(args)

def create_index(args):
    directory = os.path.join(base_path, args.dataset)
    labels = json.load(open(os.path.join(directory, "labels.json"), "r"))
    index = []
    if args.roles:
        role_index = dict([(role, []) for role in args.roles])
    else:
        role_index = defaultdict(list)

    plain_index_name = "index"
    if args.balance:
        plain_index_name += "_balance"
    if args.roles:
        plain_index_name += "_roles-" + "".join(args.roles)
    if args.cases:
        plain_index_name += "_cases-" + "".join(args.cases)
    if args.limit > 0:
        plain_index_name += f"limit-{args.limit}"

    if args.only_non_prototypical:
        dataset_directory = os.path.join("dataset_storing", args.dataset)
        word_embeddings = load_embeddings(dataset_directory, "word_embeddings")
        orig_to_bert_map = \
            pickle.load(open(os.path.join(dataset_directory, "bert_info.pkl"), "rb"))["orig_to_bert_map"]
        classifier_dir = os.path.join("classifiers", args.dataset, plain_index_name)
        classifier, labelset, labeldict = pickle.load(
            open(os.path.join(classifier_dir, f"mlp_layer-word_embeddings"), "rb"))
        A_index = labeldict["A"]
        filename = plain_index_name + "only-non-prototypical.json"
    else:
        filename = plain_index_name + ".json"

    for sent_i in range(len(labels["token"])):
        for word_i in range(len(labels["token"][sent_i])):
            role = labels["role"][sent_i][word_i]
            case = labels["case"][sent_i][word_i] if "case" in labels else None
            role_ok = args.roles is None or role in args.roles
            role_ok = role_ok and role is not None
            case_ok = args.cases is None or case in args.cases
            if role_ok and case_ok:
                if not args.only_non_prototypical or \
                   check_non_prototypical(sent_i, word_i, word_embeddings, orig_to_bert_map, classifier, A_index, role):
                    role_index[role].append((sent_i, word_i))
    if args.balance:
        min_role_len = min([len(role_index[role]) for role in role_index.keys()])
        if args.limit > 0:
            if min_role_len * len(role_index.keys()) >= args.limit:
                min_role_len = args.limit // len(role_index.keys())
            else:
                print(f"Please pick a limit which is less than the balanced length. Limit = {args.limit}, min_role_len = {min_role_len} for roles {role_index.keys()}")
                sys.exit(1)
        print(f"Culling all roles to have length {min_role_len}")
        for role in role_index.keys():
            shuffle(role_index[role])
            index.extend(role_index[role][:min_role_len])
    else:
        if args.limit > 0:
            print(f"Limit not implemented for unbalanced index yet!")
            sys.exit(1)
        for role in role_index.keys():
            index.extend(role_index[role])
    json.dump(index, open(os.path.join(directory, filename), "w"))
    print("Index has length", len(index))

def check_non_prototypical(sent_i, word_i, word_embeddings, orig_to_bert_map, classifier, A_index, role):
    bert_start_index = orig_to_bert_map[sent_i][word_i]
    word_embedding = word_embeddings[sent_i].squeeze()[bert_start_index]
    classifier_output = classifier(torch.Tensor(word_embedding))
    probs = torch.softmax(classifier_output, 0)
    A_prob = probs[A_index].item()
    if role == "A" and A_prob < 0.5:
        print("A", A_prob)
        return True
    elif role == "O" and A_prob > 0.5:
        print("O", A_prob)
        return True
    else:
        return False
    
if __name__ == "__main__":
    main()


