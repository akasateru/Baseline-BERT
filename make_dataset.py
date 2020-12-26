import csv
import os
import glob
import re
import string
import numpy as np
from keras_bert import Tokenizer
import codecs
from tqdm import tqdm

np.random.seed(0)

# 1文ずつ整形
def chenge_text(text):
    text = text.translate(str.maketrans( '', '',string.punctuation)) #特殊文字除去(@,#,$,etc.)
    text = re.sub(r'[.]{2,}','.',text) # .が2回以上続く場合,一つにまとめる
    text = re.sub(r'[	]',' ',text)
    text = re.sub(r'[ ]{2,}',' ',text)
    return text

pretrained_path = '../uncased_L-12_H-768_A-12'
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

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
        ids,masked_ids = tokenizer.encode(text[0],text[1],max_len=64)
        indices.append(ids)
        indices_mask.append(masked_ids)
    indices = np.array(indices)
    indices_mask = np.array(indices_mask)
    return [indices, indices_mask]

#class
with open('../data'+os.sep+'yahootopic'+os.sep+'classes.txt','r',encoding='utf-8',errors='ignore')as f:
    x = f.read().splitlines()
    class_0 = []
    class_1 = []
    classes = []
    for i,row in enumerate(x):
        classes.append([i,row])
        if i%2 == 0:
            class_0.append([i,row])
        elif i%2 == 1:
            class_1.append([i,row])
    
    print('class_0:',class_0)
    print('class_1:',class_1)

traindata = '../data'+os.sep+'yahootopic'+os.sep+'train_pu_half_v1.txt'
testdata = '../data'+os.sep+'yahootopic'+os.sep+'test.txt'
useclasstrain = class_1
useclasstest = classes

#traindata
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

    train = train + train_rand

    x_train = load_data(train)
    y_train = [1]*len(train_rand) + [0]*len(train_rand)
    
    print('len x_train:',len(x_train[0]))
    print('len y_train:',len(y_train))

    np.save('../dataset'+os.sep+'train'+os.sep+'x_train.npy', np.array(x_train))
    np.save('../dataset'+os.sep+'train'+os.sep+'y_train.npy', np.array(y_train))

#testdata
with open(testdata,'r',encoding='utf-8') as f:
    texts = f.read().splitlines()
    test = []
    test_label = []
    for i,text in tqdm(enumerate(texts),total=len(texts)):
        text = text.split('\t')
        for j,c in enumerate(useclasstest):
            test.append([chenge_text(text[1]),c[1]])
            if c[0] == int(text[0]):
                test_label.append(j)

    x_test = load_data(test)
    y_test = test_label

    print('len x_test:',len(x_test[0]))
    print('len y_test:',len(y_test))

    np.save('../dataset'+os.sep+'test'+os.sep+'x_test.npy', np.array(x_test))
    np.save('../dataset'+os.sep+'test'+os.sep+'y_test.npy', np.array(y_test))
