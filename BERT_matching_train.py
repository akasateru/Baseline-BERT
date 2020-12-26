import numpy as np
import os
import csv
import codecs
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras_bert.backend import keras
from keras_bert.layers import TokenEmbedding, Extract
from keras_pos_embd import PositionEmbedding
from sklearn.metrics import classification_report
from keras.utils import np_utils
from keras_position_wise_feed_forward import FeedForward
from keras_multi_head import MultiHeadAttention
from keras_transformer import gelu
from sklearn import metrics
from keras_layer_normalization import LayerNormalization
import tensorflow as tf

x_train = np.load('../dataset'+os.sep+'train'+os.sep+'x_train.npy').tolist()
y_train = np.load('../dataset'+os.sep+'train'+os.sep+'y_train.npy')
x_train =[np.array(x_train[0]),np.array(x_train[1])]

BATCH_SIZE = 32
EPOCHS = 20
LR = 5e-5
SEQ_LEN = 64
BERT_DIM = 768

pretrained_path = '../uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

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

bert_cls = bert_model.predict(x_train)
print(bert_cls.shape)

inputs = Input(shape=(768,))
output = Dense(units=1, activation='sigmoid')(inputs)
model = Model(inputs, output)
model.summary()

# lr=0.001,
model.compile(
    optimizer=Adam(beta_1=0.9,beta_2=0.999),
    loss='binary_crossentropy',
    metrics=['acc']
)

result = model.fit(
    bert_cls,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

print(result.history.keys())
import matplotlib.pyplot as plt
plt.plot(range(1,EPOCHS+1), result.history['acc'], label='acc')
plt.plot(range(1,EPOCHS+1), result.history['loss'], label='loss')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.savefig('plt.jpg')

model.save('BERT_matching_model.h5')