##### Train the probes that are used in all experiments ########
python create_dataset.py --ud-path ~/where/i/keep/ud/data/mixed_treebank-train.conllu --bert-name bert-base-uncased
python create_index.py --dataset mixed_treebank-train_bert-base-uncased --roles A O --balance
python train_classifiers.py --dataset mixed_treebank-train_bert-base-uncased --index-name index_balance_roles-AO --classifier-type mlp

##### Experiment 1 #####
python create_dataset.py --ud-path ~/where/i/keep/ud/data/en_ewt-ud-train.conllu --bert-name bert-base-uncased
python create_index.py --dataset en_ewt-ud-train_bert-base-uncased
python eval_classifiers.py --train-dataset mixed_treebank-train_bert-base-uncased --train-index index_balance_roles-AO --classifier-type mlp --eval-dataset en_ewt-ud-train_bert-base-uncased --eval-index index

##### Experiment 2 #####
## Test on original and argument-swapped versions of same sentences
python create_dataset.py --csv-file  data/argument-swapped-original.csv --bert-name bert-base-uncased
python create_index.py --dataset argument-swapped-original_bert-base-uncased
python eval_classifiers.py --train-dataset mixed_treebank-train_bert-base-uncased --train-index index_balance_roles-AO --classifier-type mlp --eval-dataset argument-swapped-original_bert-base-uncased --eval-index index

python create_dataset.py --csv-file  data/argument-swapped-swapped.csv --bert-name bert-base-uncased
python create_index.py --dataset argument-swapped-swapped_bert-base-uncased
python eval_classifiers.py --train-dataset mixed_treebank-train_bert-base-uncased --train-index index_balance_roles-AO --classifier-type mlp --eval-dataset argument-swapped-swapped_bert-base-uncased --eval-index index

##### Experiment 2.1 (local shuffling) ####
python create_dataset.py --ud-path ~/where/i/keep/ud/data/en_ewt-ud-train.conllu --bert-name bert-base-uncased --local-shuffle 3
python create_index.py --dataset en_ewt-ud-train_bert-base-uncased_localshuffle3
python eval_classifiers.py --train-dataset mixed_treebank-train_bert-base-uncased --train-index index_balance_roles-AO --classifier-type mlp --eval-dataset en_ewt-ud-train_bert-base-uncased_localshuffle3 --eval-index index
