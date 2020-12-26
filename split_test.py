with open('../data/yahootopic/test.txt','r',encoding='utf-8') as f:
    reader = f.read().splitlines()
    class_0 = []
    class_1 = []
    for row in reader:
        text = row.split('\t')
        if int(text[0])%2==0:
            class_0.append(row)
        elif int(text[0])%2==1:
            class_1.append(row)

with open('../data/yahootopic/test_v0.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(class_0))

with open('../data/yahootopic/test_v1.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(class_1))