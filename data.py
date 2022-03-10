from collections import defaultdict, Counter
import json
import numpy as np
import os
import pickle
import random
import sys
import torch
import torch.utils.data as data

from utils import load_bert_outputs, load_embeddings

class SimpleDataset(data.Dataset):
    def __init__(self, dataset_name, index_name, layer_num, old_labeldict = None, pool_method = "first"):
        self.layer_num = layer_num
        self.pool_method = pool_method
        dataset_directory = os.path.join("dataset_storing", dataset_name)
        self.labels = json.load(open(os.path.join(dataset_directory, "labels.json"), "r"))
        self.bert_info = pickle.load(open(os.path.join(dataset_directory, "bert_info.pkl"), "rb"))

        if layer_num in ["word_embeddings", "position_embeddings"]:
            self.bert_outputs = load_embeddings(dataset_directory, layer_num)
        else:
            try:
                print("in try", layer_num)
                int_layer = int(layer_num)
                self.bert_outputs = load_bert_outputs(dataset_directory, layer_num)
            except:
                print(f"Please put a valid layer name, {layer_num} is not a layer")
                sys.exit(1)
        self.index = json.load(open(os.path.join(dataset_directory, index_name + ".json"), "r"))
        print("Examples #", len(self.index))
        self.labeldict = self.get_label_dict(old_labeldict)
        self.labelset = sorted(self.labeldict.keys(), key=lambda x: self.labeldict[x])

    def __getitem__(self, idx):
        sentence_num, word_num = self.index[idx]
        bert_start_index = self.bert_info["orig_to_bert_map"][sentence_num][word_num]
        bert_end_index = self.bert_info["orig_to_bert_map"][sentence_num][word_num + 1]
        embedding = self.get_pooled_embedding(sentence_num, bert_start_index, 
                                              bert_end_index)
        role = self.labels["role"][sentence_num][word_num]
        role_label_idx = self.labeldict[role] if role in self.labeldict else -1
        aux_labels = {}
        for label_type in self.labels.keys():
            label = self.labels[label_type][sentence_num][word_num]
            if label == None:
                label = ""
            aux_labels[label_type] = label
        #aux_labels["word_index"] = word_num
        return embedding, role_label_idx, aux_labels

    # Make a labeldict of all of the labels in this dataset, keeping the same 
    # order for labels already in the old labeldict
    def get_label_dict(self, old_labeldict):
        all_labels = set()
        for sent_i, word_i in self.index:
            new_role = self.labels["role"][sent_i][word_i]
            if new_role is not None:
                all_labels.add(new_role)
        labelset = sorted(list(all_labels))
        if old_labeldict is None:
            curr_label = 0
            labeldict = {}
        else:
            labeldict = old_labeldict
            curr_label = len(old_labeldict)
        for label in labelset:
            if old_labeldict is None or label not in old_labeldict:
                labeldict[label] = curr_label
                curr_label += 1
        return labeldict

    def get_num_labels(self):
        return len(self.labeldict)

    def get_bert_dim(self):
        return self.bert_outputs[0].shape[1]

    def __len__(self):
        return len(self.index)

    def get_pooled_embedding(self, sentence_num, bert_start_index, bert_end_index):
        bert_sentence = \
            self.bert_outputs[sentence_num].squeeze()
        if self.pool_method == "first":
            return bert_sentence[bert_start_index]
        elif self.pool_method == "average":
            return np.mean(
                bert_outputs[sentence_num][self.layer_num].squeeze()\
                    [bert_start_index:bert_end_index])

    def get_dataloader(self, batch_size=32, shuffle=True):
      return data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)
