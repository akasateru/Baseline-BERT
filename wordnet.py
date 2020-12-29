from nltk.corpus import wordnet as wn
import nltk
import csv

# nltk.download("wordnet")

with open('../data/yahootopic/classes.txt','r',encoding='utf-8') as f:
    texts = f.read().splitlines()

wordnet = []
for text in texts:
    text = text.split(' & ')
    dif_all = []
    for word in text:
        word = wn.synsets(word)
        dif = word[0].definition()
        dif_all.append(dif)
    wordnet.append(' & '.join(dif_all))

print(wordnet)

with open('../data/yahootopic/classes.csv','w',encoding='utf-8') as f:
    writer = csv.writer(f)
    for t,w in zip(texts,wordnet):
        writer.writerow([t,w])
