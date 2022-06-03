import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
from keras_bert import get_custom_objects
from keras.models import load_model
from sklearn import metrics
from keras_bert import load_trained_model_from_checkpoint
from tensorflow.keras.models import Model

# パラメータ読み込み
config = json.load(open('config.json'))
BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']
LR = config['LR']
SEQ_LEN = config['SEQ_LEN']

# テストデータ読み込み
x_test = np.load('../dataset/BERT_x_test_sample.npy').tolist()
y_test = np.load('../dataset/BERT_y_test_sample.npy')
x_test =[np.array(x_test[0]),np.array(x_test[1])]

# BERTの読み込み
pretrained_path = '../uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

print("load model...")

# bert = load_trained_model_from_checkpoint(config_path,checkpoint_path,training=True,seq_len=SEQ_LEN)
# bert_nsp_dense = bert.get_layer('NSP-Dense').output
# bert_model = Model(bert.input[:2],bert_nsp_dense)
# print("bert predict...")
# bert_cls = bert_model.predict(x_test)

print("predict...")
model = load_model('../BERT_matching_model.h5',custom_objects=get_custom_objects())
pred = model.predict(x_test)

split_pred = np.array_split(pred,len(y_test))
y_pred = [np.argmax(p)+1 for p in split_pred]

labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
target_class = ["Com.","Edu.","Art.","Ath.","Off.","Mea.","Bui.","Nat.","Vil.","Ani.","Pla.","Alb.","Fil.","Wri."]
rep = metrics.classification_report(y_test,y_pred,labels=labels,target_names=target_class,digits=3)
print(rep)
with open('result.txt','w') as f:
    f.write(rep)