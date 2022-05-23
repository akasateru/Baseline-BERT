set -e
python make_dataset.py
python BERT_matching_train.py
python BERT_matching_test.py