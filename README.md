Code for running the experiments in Papadimitriou, Futrell, and Mahowald "When classifying grammatical role, BERT doesn't care about word order... except when it matters"

# How to use the code

`create_dataset.py` takes a conllu file (or a specially-formatted csv for our argument-swapped experiments), collects relevant data about each word (its grammatical role if relevant, its model-tokenized subwords and how they correspond in index) and passes all of the sentences through the desired huggingface model and saves the embeddings. Now we have a dataset of every word in our input dataset, but it's not clear which embeddings to use to train a probe

`create_index.py` creates an index: a list of locations that references the dataset. Each dataset can have multiple indices for different reason. For example, our probes are trained on a balanced set of subject and object nouns, but we might want to train a probe using all available arguments and so we can make a superset index that points to many more unbalanced examples.

`train_classifiers.py` trains a classifier using an index of locations.

`eval_classifiers.py` evals a classifier on a different dataset and index from the one it was trained on. 

Look at `sample_scripts.py` for an example of how to run the experiments in the paper.

# Data used for the paper

We used Universal Dependencies data to train and test our probes. The EWT-train set is the largest English UD file. We wanted to keep that for our evaluation, since we are focusing on evaluating our probes on rarer non-prototypical cases we wanted to make sure that they existed in the evaluation data. The EWT-test set is too small to train a probe, and so we trained our probes on a concatenation of GUM-train and EWT-test.

The argument-swapped sentences are available in the `data` directory. 
