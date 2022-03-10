
import argparse
import json
import os
import pickle

from data import SimpleDataset
from utils import get_num_layers, train_classifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--index-name', type=str)
    parser.add_argument("--classifier-type", type=str, default="mlp")
    args = parser.parse_args()
    print("args", args)

    train_classifiers(args)

def train_classifiers(args):
    classifier_dir = os.path.join("classifiers", args.dataset_name, args.index_name)
    print("making classifier dir at", classifier_dir)
    os.makedirs(classifier_dir)
    num_layers = get_num_layers(args.dataset_name)
    print(f"There are {num_layers} layers in this model")
    layers = ["word_embeddings"] +  [str(i) for i in range(num_layers + 1)]
    logistic = args.classifier_type == "logistic"
    for layer in layers:
        print(f"Layer {layer}")
        dataset = SimpleDataset(args.dataset_name, args.index_name, layer)
        classifier = train_classifier(dataset, logistic)
        pickle.dump((classifier, dataset.labelset, dataset.labeldict),
            open(os.path.join(classifier_dir, f"{args.classifier_type}_layer-{layer}"), "wb"))

main()
