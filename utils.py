from collections import defaultdict
import conllu 
import csv
import h5py
import numpy as np
import os
import random
import sys
import torch
from tqdm import tqdm
from transformers import BertModel

def get_tokens_and_labels(filename):
    """
    Parameters:
    filename: te location of the treebank (conll file)

    This function parses the conll file to get:
    - labels: A dict, whose keys are types of labels (eg, "animacy"), and each 
        value is a list of length num_sentences
    """
    with open(filename) as f:
        conll_data = f.read()
    sentences = conllu.parse(conll_data)
    labels = defaultdict(list)
    num_nouns = 0
    num_relevant_examples = 0
    for sent_i, tokenlist in enumerate(sentences):
        sentence_info = defaultdict(list)
        if "sent_id" in tokenlist.metadata.keys():
            sentence_info["sent_id"] = [tokenlist.metadata["sent_id"]]*len(tokenlist)
        noun_count = 0
        for token in tokenlist:
            token_info = get_token_info(token, tokenlist)
            token_case = None
            token_animacy = ""
            if token_info["role"] is not None:
                if token['feats'] and 'Case' in token['feats']:
                    token_case = token['feats']['Case']
                if token['feats'] and 'Animacy' in token['feats']:
                    token_animacy = token['feats']['Animacy']
            token_info["case"] = token_case
            token_info["animacy"] = token_animacy
            sentence_info["token"].append(token['form'])
            for label_type in token_info.keys():
                sentence_info[label_type].append(token_info[label_type])
            sentence_info["preceding_nouns"].append(noun_count)
            if token["upostag"] == "NOUN" or token["upostag"] == "PROPN" or token["upostag"]=="PRON":
                noun_count += 1
        for label_type in sentence_info.keys():
            labels[label_type].append(sentence_info[label_type])
        labels["word_index"].append(list(range(len(sentence_info["token"]))))
        assert len(sentence_info["case"]) == len(sentence_info["role"]), \
               "Length of case and role should be the same for every sentence (though both lists can include Nones)"
    print("returning from get_tokens, the keys are", list(labels.keys()))
    return dict(labels)

def get_tokens_and_labels_csv(filename):
    labels = defaultdict(list)
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            sentence = row['sentence'].split(" ")
            labels["token"].append(sentence)
            labels['sent_id'].append([row['sentence_id']]*len(sentence))
            subject_idx = int(row["subject_idx"])
            object_idx = int(row["object_idx"])
            roles = [None]*len(sentence)
            roles[subject_idx] = "A"
            roles[object_idx] = "O"
            labels["role"].append(roles)
            subject_words = [None]*len(sentence)
            subject_words[subject_idx] = row["subject"]
            subject_words[object_idx] = row["subject"]
            labels["subject_word"].append(subject_words)
            object_words = [None]*len(sentence)
            object_words[subject_idx] = row["object"]
            object_words[object_idx] = row["object"]
            labels["object_word"].append(object_words)
            verb_words = [None]*len(sentence)
            verb_words[subject_idx] = row["verb"]
            verb_words[object_idx] = row["verb"]
            labels["verb_word"].append(verb_words)
            labels["word_index"].append(list(range(len(sentence))))
    return labels

def get_token_info(token, tokenlist):
    token_info = {}
    token_info["role"] = None
    token_info["verb_word"] = ""
    token_info["verb_idx"] = -1
    token_info["subject_word"] = ""
    token_info["object_word"] = ""
    if not (token["upostag"] == "NOUN" or token["upostag"] == "PROPN"):
        return token_info

    head_id = token['head']
    head_list = tokenlist.filter(id=head_id)
    head_pos = None
    if len(head_list) > 0:
        head_token = head_list[0]
        if head_token["upostag"] == "VERB":
            head_pos = "verb"
            token_info["verb_word"] = head_token["lemma"]
            token_info["verb_idx"] = int(head_token["id"]) - 1
        elif head_token["upostag"] == "AUX":
            head_pos = "aux"
            token_info["verb_word"] = head_token["lemma"]
            token_info["verb_idx"] = int(head_token["id"]) - 1
        else:
            return token_info

    if "nsubj" in token['deprel']:
        token_info["subject_word"] = token['form']
        has_object = False
        has_expletive_sibling = False
        # 'deps' field is often empty in treebanks, have to look through
        # the whole sentence to find if there is any object of the head
        # verb of this subject (this would determine if it's an A or an S)
        for obj_token in tokenlist:
            if obj_token['head'] == head_id:
                if "obj" in obj_token['deprel']:
                    has_object = True
                    token_info["object_word"] = obj_token["form"]
                if obj_token['deprel'] == "expl":
                    has_expletive_sibling = True
        if has_expletive_sibling:
            token_info["role"] = "S-expletive"
        elif has_object:
            token_info["role"] = "A"
        else:
            token_info["role"] = "S"
        if "pass" in token['deprel']:
            token_info["role"] += "-passive"
    elif "obj" in token['deprel']:
        token_info["role"] = "O"
        token_info["object_word"] = token['form']
        for subj_token in tokenlist:
            if subj_token['head'] == head_id:
                if "subj" in subj_token['deprel']:
                    token_info["subject_word"] = subj_token['form']
    if head_pos == "aux" and token_info["role"] is not None:
        token_info["role"] += "-aux"
    return token_info

def get_bert_tokens(orig_tokens, tokenizer):
    """
    Given a list of sentences, return a list of those sentences in BERT tokens,
    and a list mapping between the indices of each sentence, where
    bert_tokens_map[i][j] tells us where in the list bert_tokens[i] to find the
    start of the word in sentence_list[i][j]
    The input orig_tokens should be a list of lists, where each element is a word.
    """
    bert_tokens = []
    orig_to_bert_map = []
    bert_to_orig_map = []
    for i, sentence in enumerate(orig_tokens):
        sentence_bert_tokens = []
        sentence_map_otb = []
        sentence_map_bto = []
        sentence_bert_tokens.append("[CLS]")
        for orig_idx, orig_token in enumerate(sentence):
            sentence_map_otb.append(len(sentence_bert_tokens))
            tokenized = tokenizer.tokenize(orig_token)
            for bert_token in tokenized:
                sentence_map_bto.append(orig_idx)
            sentence_bert_tokens.extend(tokenizer.tokenize(orig_token))
        sentence_map_otb.append(len(sentence_bert_tokens))
        sentence_bert_tokens = sentence_bert_tokens[:511]
        sentence_bert_tokens.append("[SEP]")
        bert_tokens.append(sentence_bert_tokens)
        orig_to_bert_map.append(sentence_map_otb)
        bert_to_orig_map.append(sentence_map_bto)
    bert_ids = [tokenizer.convert_tokens_to_ids(b) for b in bert_tokens]
    return bert_tokens, bert_ids, orig_to_bert_map, bert_to_orig_map

def shuffle_positions(labels, shuffle_positions, local_shuffle):
    if not shuffle_positions and local_shuffle <= 0:
        print("Not shuffling positions this time")
        return labels
    assert not shuffle_positions or local_shuffle <= 0, \
        "Must choose between local and global shuffling!"
    labels["shuffled_index"] = []

    for sent_i, sentence in enumerate(labels["token"]):
        length = len(sentence)
        if shuffle_positions:
            permutation = list(range(length))
            random.shuffle(permutation)
        elif local_shuffle > 0:
            permutation = list(range(length))
            for chunk_start in range(0, length, local_shuffle):
                chunk_end = min(chunk_start + local_shuffle, length)
                chunk = permutation[chunk_start:chunk_end]
                random.shuffle(chunk)
                permutation[chunk_start:chunk_end] = chunk
        for label in labels:
            if label is not "shuffled_index":
                labels[label][sent_i] = \
                    [labels[label][sent_i][permutation[word_i]] for word_i in range(length)]
        labels["shuffled_index"].append(list(range(length)))
    return labels

def save_sample(num_samples, labels, directory):
    samples = []
    sampled_sentences = random.sample(range(len(labels['token'])), num_samples)
    for sent_i in sampled_sentences:
        sentence = " ".join(labels["token"][sent_i])
        sentence_id = labels["sent_id"][sent_i][0]
        samples.append([sentence_id, sentence])
    with open(os.path.join(directory, "sample.csv"), "w") as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["sentence_id", "sentence"])
        writer.writerows(samples)

def save_bert_outputs(directory, bert_ids, bert_model, shuffle_positions=False, single_position=-1):
    """
    Given a list of lists of bert IDs, runs them through BERT.
    Cache the results to hdf5_path, and load them from there if available.
    """
    assert not shuffle_positions or not single_position >= 0, \
        "Choose beetween shuffling and putting a single position"

    datafile = h5py.File(os.path.join(directory, "bert_vectors.hdf5"), 'w')
    word_file = h5py.File(os.path.join(directory, "bert_word_embs.hdf5"), 'w')
    position_file = h5py.File(os.path.join(directory, "bert_position_embs.hdf5"), 'w')
    with torch.no_grad():
        print(f"Running {len(bert_ids)} sentences through BERT. This takes a while")
        for idx, sentence in enumerate(tqdm(bert_ids)):
            if single_position >= 0:
                positions = torch.ones((1, len(sentence)), dtype=torch.long) * single_position
            else:  
                positions = torch.tensor(range(len(sentence))).unsqueeze(0)

            bert_output = bert_model(torch.tensor(sentence).unsqueeze(0),
                                     position_ids = positions)
            hidden_layers = bert_output["hidden_states"]
            layer_count = len(hidden_layers)
            _, sentence_length, dim = hidden_layers[0].shape
            dset = datafile.create_dataset(str(idx), (layer_count, sentence_length, dim))
            dset[:, :, :] = np.vstack([np.array(x) for x in hidden_layers])

            word_embedding = bert_model.embeddings.word_embeddings(torch.tensor(sentence))
            sentence_length, dim = word_embedding.shape
            word_dset = word_file.create_dataset(str(idx), (sentence_length, dim))
            word_dset[:,:] = word_embedding

            position_embedding = bert_model.embeddings.position_embeddings(positions)
            pos_dset = position_file.create_dataset(str(idx), (sentence_length, dim))
            pos_dset[:,:] = position_embedding
    datafile.close()
    word_file.close()
    position_file.close()

def save_just_position_word(directory, bert_ids, bert_model, shuffle_positions=False, single_position=-1):
    """
    NOTE: NOT USED. save_bert_outputs includes this functionality.
    Given a list of lists of bert IDs, runs them through BERT.
    Cache the results to hdf5_path, and load them from there if available.
    """
    assert not shuffle_positions or not single_position >= 0, \
        "Choose beetween shuffling and putting a single position"

    word_file = h5py.File(os.path.join(directory, "bert_word_embs.hdf5"), 'w')
    position_file = h5py.File(os.path.join(directory, "bert_position_embs.hdf5"), 'w')
    with torch.no_grad():
        print(f"Running {len(bert_ids)} sentences through BERT. This takes a while")
        for idx, sentence in enumerate(tqdm(bert_ids)):
            if single_position >= 0:
                positions = torch.ones((1, len(sentence)), dtype=torch.long) * single_position
            elif shuffle_positions:
                # Shuffle positions of everything except for first and last BERT tokens.
                positions = torch.arange(len(sentence), dtype=torch.long)
                permutation = torch.randperm(len(sentence)-2)
                positions[1:-1] = positions[1:-1][permutation]
                positions = positions.unsqueeze(0)
            else:  
                positions = torch.tensor(range(len(sentence))).unsqueeze(0)

            word_embedding = bert_model.embeddings.word_embeddings(torch.tensor(sentence))
            sentence_length, dim = word_embedding.shape
            word_dset = word_file.create_dataset(str(idx), (sentence_length, dim))
            word_dset[:,:] = word_embedding

            position_embedding = bert_model.embeddings.position_embeddings(positions)
            pos_dset = position_file.create_dataset(str(idx), (sentence_length, dim))
            pos_dset[:,:] = position_embedding
    word_file.close()
    position_file.close()

def load_bert_outputs(directory, layer):
    hdf5_path = os.path.join(directory, "bert_vectors.hdf5")
    try:
        layer = int(layer)
    except:
        print("Please use a valid layer in 0-12. If you want word embeddings, use the get_word_embeddings method")
    outputs = []
    try:
        with h5py.File(hdf5_path, 'r') as datafile:
            max_key = max([int(key) for key in datafile.keys()])
            for i in tqdm(range(max_key + 1), desc='[Loading from disk]'):
                hidden_layers = datafile[str(i)[:]]
                output = np.array(hidden_layers[layer])
                outputs.append(output)
            print(f"Loaded {i} sentences from disk.")
    except OSError:
        print(f"Encountered hdf5 reading error on file {hdf5_path}. Please re-create the hdf5 file")
        sys.exit(1)
    return outputs

def load_embeddings(directory, embeddings_type):
    if embeddings_type == "word_embeddings":
        hdf5_path = os.path.join(directory, "bert_word_embs.hdf5")
    elif embeddings_type == "position_embeddings":
        hdf5_path = os.path.join(directory, "bert_position_embs.hdf5")
    else:
        print(embeddings_type, "Is not not word_embeddings or position_embeddings")
        sys.exit(1)
    
    outputs = []
    try:
        with h5py.File(hdf5_path, 'r') as datafile:
            max_key = max([int(key) for key in datafile.keys()])
            for i in tqdm(range(max_key + 1), desc='[Loading from disk]'):
                outputs.append(np.array(datafile[str(i)][:]))
    except OSError:
        print(f"Encountered hdf5 reading error on file {hdf5_path}. Please re-create the hdf5 file")
        sys.exit(1)
    return outputs

def get_num_layers(dataset_name):
    dataset_directory = os.path.join("dataset_storing", dataset_name)
    hdf5_path = os.path.join(dataset_directory, "bert_vectors.hdf5")
    try:
        with h5py.File(hdf5_path, 'r') as datafile:
            hidden_layers = datafile[str(0)[:]]
            return hidden_layers.shape[0]
    except OSError:
        print(f"Encountered hdf5 reading error on file {hdf5_path}. Please re-create the hdf5 file")
        sys.exit(1)

class _classifier(nn.Module):
    def __init__(self, nlabel, bert_dim):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(bert_dim, 64),
            nn.ReLU(),
            nn.Linear(64, nlabel),
            nn.Dropout(.1)
        )
    def forward(self, input):
        return self.main(input)

def train_classifier(dataset, logistic):                                        
    if logistic:                                                                
        return train_classifier_logistic(dataset)                               
    else:                                                                       
        return train_classifier_mlp(dataset)                                    

def train_classifier_mlp(train_dataset, epochs=20):
    classifier = _classifier(train_dataset.get_num_labels(), train_dataset.get_bert_dim())
    optimizer = optim.Adam(classifier.parameters())
    criterion = nn.CrossEntropyLoss()

    dataloader = train_dataset.get_dataloader()

    for epoch in range(epochs):
        losses = []
        for emb_batch, role_label_batch, _ in dataloader:
            output = classifier(emb_batch)
            loss = criterion(output, role_label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean().item())
        print('[%d/%d] Train loss: %.3f' % (epoch+1, epochs, np.mean(losses)))
    return classifier

def train_classifier_logistic(train_dataset):                                   
    X, y = [], []                                                               
    dataloader = train_dataset.get_dataloader(batch_size=1)                     
    for emb_batch, role_label_batch, _ in dataloader:                           
        X.append(emb_batch[0])                                                  
        y.append(role_label_batch[0])                                           
    X = np.stack(X, axis=0)                                                     
    y = np.stack(y, axis=0)                                                     
    scaler = preprocessing.StandardScaler().fit(X)                              
    X_scaled = scaler.transform(X)                                              
    classifier = LogisticRegression(random_state=0, max_iter=10000).fit(X_scaled, y)
    return classifier                                                           
