import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from data_gen import DataGenerator
from config import Config
import data_gen
from pre_recall import *


root_data_path='data_files'
data_gen_obj=DataGenerator(root_data_path,temporal_stride=4,temporal_length=16,resize=224)


train_data = data_gen_obj.load_samples(data_cat='train')
test_data = data_gen_obj.load_samples(data_cat='test')


print('num of train_samples: {}'.format(len(train_data)))
print('num of test_samples: {}'.format(len(test_data)))

train_generator = data_gen_obj.data_generator(train_data,batch_size=4,shuffle=True)
test_generator = data_gen_obj.data_generator(test_data,batch_size=4,shuffle=True)

import keras
from keras.layers.recurrent import LSTM
from keras.models import Input,Model,InputLayer
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, BatchNormalization
import tensorflow as tf
from keras import layers
from keras.applications import InceptionV3



def get_model(num_classes=Config.num_classes):
    # Define model

    video = keras.Input(shape = (16,224,224,3), name='video')
    cnn = InceptionV3(weights='imagenet',include_top=False, pooling='avg')
    cnn.trainable =False
    
    frame_features = layers.TimeDistributed(cnn)(video)
    blstm_1 = Bidirectional(LSTM(1024, dropout=0.1, recurrent_dropout=0.5, return_sequences= True))(frame_features)
    blstm_2 = Bidirectional(LSTM(1024, dropout=0.1, recurrent_dropout=0.5, return_sequences= False))(blstm_1)
    Dense_2   = Dense(256, activation = 'sigmoid' )(blstm_2)
    batchNorm = BatchNormalization()(Dense_2)
    enver   = Dense(32, activation = 'sigmoid')(batchNorm)
    batchNorm2= BatchNormalization()(enver)
    Dense_3   = Dense(num_classes, activation='sigmoid')(batchNorm2)
    model = keras.models.Model(input = video , output = Dense_3)


    model.summary()
    #plot_model(model, show_shapes=True,
    #           to_file='model.png')

    from keras.optimizers import SGD
    sgd = SGD(lr=0.002, decay = 1e-5, momentum=0.9, nesterov=True)
    
    model.compile(loss = 'categorical_crossentropy',  optimizer=sgd, metrics=['acc',f1_m,precision_m, recall_m])
    return model

model = get_model()

# Fit model using generator
hist = model.fit_generator(train_generator,
                steps_per_epoch=len(train_data),epochs=3,
                validation_data=test_generator,
                validation_steps=len(test_data))


loss, accuracy, f1_score, precision, recall = model.evaluate(X_valid, Y_valid, verbose=0)


from sklearn.metrics import classification_report
y_pred = model.predict(X_valid, batch_size=16, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(Y_valid, y_pred_bool))



from sklearn.metrics import roc_curve,roc_auc_score
fpr , tpr , thresholds = roc_curve ( Y_val , y_pred)
def plot_roc_curve(fpr,tpr):
  plt.plot(fpr,tpr)
  plt.axis([0,1,0,1])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.show()

plot_roc_curve (fpr,tpr)
auc_score=roc_auc_score(Y_valid, y_pred)



