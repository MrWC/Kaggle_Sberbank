
# coding: utf-8

# In[ ]:

import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
import numpy as np
import datetime, time, json

from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Merge, BatchNormalization, Activation, Convolution1D, MaxPooling1D
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend
from keras import initializers
from keras.optimizers import SGD


# In[ ]:

train_df = pd.read_csv("train_df_after_processing.csv")


# In[ ]:

y_train = np.array(train_df['price_doc'])
x_train = np.array(train_df.drop(['id','price_doc'],axis=1))


# In[ ]:

dropout = 0.5
weights = initializers.RandomUniform(minval=-0.05, maxval=1, seed=2)
bias = bias_initializer='zeros'

model = Sequential()
model.add(Dense(128, kernel_initializer=weights, bias_initializer=bias,input_shape=(x_train.shape[1],)))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(dropout))

model.add(Dense(64, kernel_initializer=weights, bias_initializer=bias))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(dropout))
'''
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(dropout))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(dropout))
'''
model.add(Dense(32))

model.add(Dense(16))

model.add(Dense(8))

model.add(Dense(1))


# In[ ]:

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

save_best_weights = 'question_pairs_weights.h5'

t0 = time.time()
callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
             EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')]
history = model.fit(x_train[:,:],
                    y_train[:],
                    batch_size=32,
                    epochs=100, #Use 100, I reduce it for Kaggle,
                    validation_split=0.15,
                    verbose=True,
                    shuffle=True,
                    callbacks=callbacks)
t1 = time.time()
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


# In[ ]:

# Make predictions with the best weights
model.load_weights(save_best_weights)
predictions = model.predict(x_train[:,:], verbose = True)

pd.DataFrame(predictions).describe()


