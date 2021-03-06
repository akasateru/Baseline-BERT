import csv
import os
import glob
import re
import string
import numpy as np
from keras_bert import Tokenizer
import codecs
from tqdm import tqdm
import json

np.random.seed(0)

json_file = open('config.json','r')
config = json.load(json_file)
SEQ_LEN = config['SEQ_LEN']

pretrained_path = '../uncased_L-12_H-768_A-12'
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# 1文ずつ整形
def chenge_text(text):
    text = text.translate(str.maketrans( '', '',string.punctuation)) #特殊文字除去 !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    text = re.sub(r'[ ]{2,}',' ',text)
    return text

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# texts:[文章, クラス名]
def load_data(texts):
    tokenizer = Tokenizer(token_dict)
    indices = []
    indices_mask = []
    for text in tqdm(texts):
        ids,masked_ids = tokenizer.encode(text[0],text[1],max_len=SEQ_LEN)
        indices.append(ids)
        indices_mask.append(masked_ids)
    indices = np.array(indices)
    indices_mask = np.array(indices_mask)
    return [indices, indices_mask]

# yahoo topic class
# classes.csv
#   ['class名','wordnet']
with open('../data/yahootopic/classes.csv','r',encoding='utf-8',errors='ignore')as f:
    reader = csv.reader(f)
    yahoo_class = []
    for i,row in enumerate(reader):
        yahoo_class.append([i,row[0]])

#dbpedia class
db_class = []
with open('../data/dbpedia/dbpedia_csv/classes.txt','r',encoding='utf-8') as f:
    reader = f.read().splitlines()
    for i,r in enumerate(reader):
        db_class.append([i,'this text is about ' + r])

traindata = '../data/yahootopic/train.txt'
testdata = '../data/dbpedia/dbpedia_csv/test.csv'
useclasstrain = yahoo_class
useclasstest = db_class

# traindata
with open(traindata,'r',encoding='utf-8') as f:
    texts = f.read().splitlines()
    train = []
    train_rand = []
    for i,text in tqdm(enumerate(texts),total=len(texts)):
        text = text.split('\t')
        for c in useclasstrain:
            if c[0] == int(text[0]):
                train.append([chenge_text(text[1]),c[1]])
                break
        rand_base = [c[1] for c in useclasstrain]
        rand_base.remove(train[i][1])
        rand = np.random.choice(rand_base)
        train_rand.append([chenge_text(text[1]),rand])

    train_data = train + train_rand
    x_train = load_data(train_data)
    y_train = [1]*len(train) + [0]*len(train_rand)

    print('len x_train:',len(x_train[0]))
    print('len y_train:',len(y_train))

    np.save('../dataset/train/x_train.npy', np.array(x_train))
    np.save('../dataset/train/y_train.npy', np.array(y_train))

with open(testdata,'r',encoding='utf-8') as f:
    reader = csv.reader(f)
    x_test = []
    y_test = []
    test = []
    test_label = []
    l = len(list(reader))
    f.seek(0)
    for row in tqdm(reader,total=l):
        text_stock = []
        text = row[2].replace('(',')').split(')')
        for i,t in enumerate(text):
            if i % 2 == 0:
                text_stock.append(t)
        text = ''.join(text_stock)
        text = chenge_text(text)
        text = ' '.join([x for x in text.split(' ') if x not in row[1].split(' ')])
        text = text.replace('  ',' ')
        for j,c in enumerate(useclasstest):
            test.append([chenge_text(text),c[1]])
            if c[0] == int(row[0])-1:
                test_label.append(j)
    x_test = load_data(test)
    y_test = test_label

print('len x_test:',len(x_test[0]))
print('len y_test:',len(y_test))

np.save('../dataset/test/x_test.npy', np.array(x_test))
np.save('../dataset/test/y_test.npy', np.array(y_test))