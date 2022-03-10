
import argparse
from collections import defaultdict
import datetime
import json
import numpy as np
import os
import pandas as pd
import pickle
import torch

from data import SimpleDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset', type=str)
    parser.add_argument('--train-index', type=str)
    parser.add_argument("--classifier-type", type=str, default="mlp")
    parser.add_argument('--eval-dataset', type=str)
    parser.add_argument('--eval-index', type=str)
    args = parser.parse_args()
    print("args", args)

    eval_classifiers(args)


def eval_classifiers(args):
    classifier_dir = os.path.join("classifiers", args.train_dataset, args.train_index)
    print("Evaluating classifiers at", classifier_dir)
    layers = ["position_embeddings", "word_embeddings"] +  [str(i) for i in range(13)]
    logistic = args.classifier_type == "logistic"
    results = defaultdict(list)
    for layer in layers:
        classifier, labelset, labeldict = pickle.load(
            open(os.path.join(classifier_dir, f"{args.classifier_type}_layer-{layer}"), "rb"))
        print("Classifier labeldict", labeldict)
        A_index = labeldict["A"]
        eval_dataset = SimpleDataset(args.eval_dataset, args.eval_index, layer, old_labeldict = labeldict)
        dataloader = eval_dataset.get_dataloader(batch_size=1)
        classifier.eval()
        for embedding, role, other_labels in dataloader:
            if args.classifier_type == "logistic":
                probs = classifier.predict_proba(torch.Tensor(embedding))[0]
                A_prob = probs[A_index]
            elif args.classifier_type == "mlp":
                output = classifier(torch.Tensor(embedding))
                probs = torch.softmax(output, 1)
                A_prob = probs[:,A_index][0].item()
            for label in other_labels.keys():
                val = other_labels[label][0]
                if type(val) == torch.Tensor:
                    val = val.item()
                results[label].append(val)
            results["layer"].append(layer)
            results["probability_A"].append(A_prob)
    df = pd.DataFrame(results)
    date_string = datetime.datetime.now().strftime("%m%d%Y")
    output_file = f"{date_string}_train-{args.train_dataset}-{args.train_index}_eval-{args.eval_dataset}-{args.eval_index}.csv"
    df.to_csv(open(os.path.join("results", "long_names", output_file), "w"))
    

def add_classifier_predictions(labels_file, vecs_file, classifier, label_dict, layer, classifier_type):
    vecs = h5py.File(vecs_file, "r")[f"bert_layer_{layer}"]
    labels = json.load(open(labels_file, "r"))
    print("after loading", labels.keys())
    length = len(vecs)
    labels[f"probA_{classifier_type}_layer_{layer}"] = [0]*length
    A_index = label_dict["A"]
    for i in range(length):
        if classifier_type == "logistic":
            probs = classifier.predict_proba(torch.Tensor(vecs[i].astype(np.float32)).unsqueeze(0))[0]
            A_prob = probs[A_index]
        elif classifier_type == "mlp":
            output = classifier(torch.Tensor(vecs[i].astype(np.float32)).unsqueeze(0))
            probs = torch.softmax(output, 1)
            A_prob = probs[:,A_index][0].item()
        labels[f"probA_{classifier_type}_layer_{layer}"][i] = A_prob
    print("before saving", labels.keys())
    json.dump(labels, open(labels_file, "w"))


if __name__ == "__main__":
    main()
