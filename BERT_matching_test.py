import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
from keras_bert import get_custom_objects
from keras.models import load_model
from sklearn import metrics

# パラメータ読み込み
config = json.load(open('config.json'))
BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']
LR = config['LR']
SEQ_LEN = config['SEQ_LEN']

# テストデータ読み込み
x_test = np.load('../dataset/BERT_x_test.npy').tolist()
y_test = np.load('../dataset/BERT_y_test.npy')
x_test =[np.array(x_test[0]),np.array(x_test[1])]

print("load model...")
model = load_model('../BERT_matching_model.h5',custom_objects=get_custom_objects())

print("predict...")
pred = model.predict(x_test)
y_pred = np.array(np.array_split(pred,len(y_test))).argmax(axis=1)

rep = metrics.classification_report(y_test,y_pred,digits=3)
print(rep)
with open('result.txt','w') as f:
    f.write(rep)