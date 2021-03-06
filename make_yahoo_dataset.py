from nltk.corpus import wordnet as wn
import nltk
import csv

# クラスの整形
nltk.download("wordnet")

with open('../data/yahootopic/classes.txt','r',encoding='utf-8') as f:
    texts = f.read().splitlines()

wordnet = []
for text in texts:
    text = text.split(' & ')
    dif_all = []
    for word in text:
        word = wn.synsets(word)
        dif = word[0].definition()
        dif = 'this text is about '+dif+' .'
        dif_all.append(dif)
    wordnet.append(' '.join(dif_all))

classes = []
for text in texts:
    text = text.split(' & ')
    dif_all = []
    for word in text:
        dif = 'this text is about '+word+' .'
        dif_all.append(dif)
    classes.append(' '.join(dif_all))

with open('../data/yahootopic/classes.csv','w',encoding='utf-8',newline='') as f:
    writer = csv.writer(f)
    for t,w in zip(classes,wordnet):
        writer.writerow([t,w])

# 学習データのv0とv1を連結
with open('../data/yahootopic/train_pu_half_v0.txt','r',encoding='utf-8') as f:
    text_v0 = f.read()

with open('../data/yahootopic/train_pu_half_v1.txt','r',encoding='utf-8') as f:
    text_v1 = f.read()

text = text_v0+text_v1

with open('../data/yahootopic/train.txt','w',encoding='utf-8') as f:
    f.write(text)


