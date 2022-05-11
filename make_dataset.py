import csv
import re
import string
import json
import numpy as np
from keras_bert import Tokenizer
from tqdm import tqdm
from nltk.corpus import wordnet

np.random.seed(0)
config = json.load(open('config.json','r'))
SEQ_LEN = config['SEQ_LEN']

with open('../uncased_L-12_H-768_A-12/vocab.txt','r',encoding='utf8') as f:
    token_dict = {token:i for i,token in enumerate(f.read().splitlines())}
tokenizer = Tokenizer(token_dict)

# 前処理
def preprocessing(text,auth):
    # 括弧内文章の削除
    text = re.sub(r'\(.*?\)','',text)
    # 記号文字の削除
    text = text.translate(str.maketrans('','',string.punctuation))
    # 著者名の削除
    text = text.replace(auth,'')
    # スペースの調整
    text = re.sub(r'\s+',' ',text)
    return text

# preprocessing train data -----------------------------------------------------------------------
# load topic class labels
with open('../data/topic/classes.txt','r',encoding='utf-8') as f:
    labels = f.read().splitlines()
topic_class_hypothesis = dict()
for i,label in enumerate(labels):
    topic_class_hypothesis[i] = 'this text is about ' + ' or '.join([wordnet.synsets(word)[0].definition() for word in label.split(' & ')])

# load train data
with open('../data/topic/train_pu_half_v0.txt','r',encoding='utf-8') as f:
    texts_v0 = f.read()
with open('../data/topic/train_pu_half_v1.txt','r',encoding='utf-8') as f:
    texts_v1 = f.read()
texts = texts_v0 + texts_v1

y_train = []
indeces, segments = [],[]
for label_text in tqdm(texts.splitlines()):
    label,text = label_text.split('\t')
    rand_base = [0,1,2,3,4,5,6,7,8,9]
    rand_base.remove(int(label))
    label_rand = np.random.choice(rand_base)
    text = preprocessing(text,'')
    ids, segs = tokenizer.encode(first=text, second=topic_class_hypothesis[int(label)], max_len=SEQ_LEN)
    indeces.append(ids)
    segments.append(segs)
    y_train.append(1)
    ids, segs = tokenizer.encode(first=text, second=topic_class_hypothesis[int(label_rand)], max_len=SEQ_LEN)
    indeces.append(ids)
    segments.append(segs)
    y_train.append(0)
x_train = [np.array(indeces),np.array(segments)]

np.save('../dataset/BERT_x_train.npy', x_train)
np.save('../dataset/BERT_y_train.npy', y_train)

# dbpedia class ------------------------------------------------------------------------------------------------------
with open('../data/dbpedia_csv/classes.txt','r',encoding='utf-8') as f:
    dbpedia_class = { i+1:'this text is about '+text for i,text in enumerate(f.read().splitlines())}

with open('../data/dbpedia_csv/test.csv','r',encoding='utf-8') as f:
    reader = csv.reader(f)
    y_test = []
    indeces, segments = [],[]
    for cls_num,auth,readtext in reader:
        ids, segs = tokenizer.encode(first=preprocessing(readtext,auth), second=dbpedia_class[int(cls_num)], max_len=SEQ_LEN)
        indeces.append(ids)
        segments.append(segs)
        y_test.append(int(cls_num))
    x_test = [np.array(indeces),np.array(segments)]

np.save('../dataset/BERT_x_test.npy', x_test)
np.save('../dataset/BERT_y_test.npy', y_test)