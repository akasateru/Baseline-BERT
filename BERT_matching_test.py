import numpy as np
import os
import json
from keras_bert import load_trained_model_from_checkpoint
from keras.models import Model, load_model
from keras_bert.layers import TokenEmbedding, Extract
from keras_pos_embd import PositionEmbedding
from keras_position_wise_feed_forward import FeedForward
from keras_multi_head import MultiHeadAttention
from keras_transformer import gelu
from sklearn import metrics
from keras_layer_normalization import LayerNormalization

# パラメータ読み込み
json_file = open('config.json')
config = json.load(json_file)
BATCH_SIZE = config['BATCH_SIZE']
EPOCHS = config['EPOCHS']
LR = config['LR']
SEQ_LEN = config['SEQ_LEN']

# BERT読み込み
pretrained_path = '../uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# テストデータ読み込み
x_test = np.load('../dataset/test/x_test.npy').tolist()
y_test = np.load('../dataset/test/y_test.npy')
x_test =[np.array(x_test[0]),np.array(x_test[1])]

bert = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN
)

inputs = bert.inputs[:2]
dense = bert.get_layer('NSP-Dense').output
bert_model = Model(inputs, dense)

x_test = bert_model.predict(x_test)
print(x_test.shape)

custom_objects = {
'TokenEmbedding': TokenEmbedding,
'PositionEmbedding': PositionEmbedding,
'MultiHeadAttention': MultiHeadAttention,
'FeedForward': FeedForward,
'gelu': gelu,
'Extract': Extract,
'LayerNormalization': LayerNormalization
}

model = load_model('BERT_matching_model.h5',custom_objects=custom_objects)

y_pred = model.predict(x_test)
y_pred = np.array_split(y_pred,len(y_test))
y_pred_list = []
for y_p in y_pred:
    y_pred_list.append(np.argmax(y_p))

y_pred = y_pred_list
rep = metrics.classification_report(y_test,y_pred,digits=3)
print(rep)
with open('result.txt','w') as f:
    f.write(rep)


