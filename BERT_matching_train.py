import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
from keras_bert import load_trained_model_from_checkpoint
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
from keras_bert import AdamWarmup, calc_train_steps

# パラメータの読み込み
config = json.load(open('config.json'))
BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']
SEQ_LEN = config['SEQ_LEN']
LR = config['LR']

# 学習データ読み込み
print("load data...")
x_train = np.load('../dataset/BERT_x_train_sample.npy').tolist()
y_train = np.load('../dataset/BERT_y_train_sample.npy')
x_train =[np.array(x_train[0]),np.array(x_train[1])]

# BERTの読み込み
pretrained_path = '../uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

print("load pretrained model...")
bert = load_trained_model_from_checkpoint(config_path,checkpoint_path,training=True,seq_len=SEQ_LEN)

decay_steps, warmup_steps = calc_train_steps(x_train[0].shape[0],batch_size=BATCH_SIZE,epochs=EPOCHS)

# bert_nsp_dense = bert.get_layer('NSP-Dense').output
# bert_model = Model(bert.input[:2],bert_nsp_dense)
# print("bert predict...")
# bert_cls = bert_model.predict(x_train)

# inputs = Input(shape=(768,))
# output = Dense(units=1, activation='sigmoid')(inputs)
# model = Model(inputs,output)

inputs = bert.input[:2]
bert_nsp_dense = bert.get_layer('NSP-Dense').output
outputs = Dense(units=1, activation='sigmoid')(bert_nsp_dense)
model = Model(inputs,outputs)
model.compile(optimizer=AdamWarmup(decay_steps=decay_steps,warmup_steps=warmup_steps,learning_rate=LR),loss='binary_crossentropy',metrics=['mae','mse','acc'])
print("training...")
result = model.fit(bert_cls,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE)

pred = model.predict(x_train)
print(pred)

print("save...")
model.save('../BERT_matching_model.h5')