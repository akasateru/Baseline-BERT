from nltk.corpus import wordnet as wn
import nltk
import csv

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
