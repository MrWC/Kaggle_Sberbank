
# coding: utf-8

# In[1]:

import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pandas as pd
import numpy as np
import datetime, time, json
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Merge, BatchNormalization, Activation, Convolution1D, MaxPooling1D
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend
from keras import initializers
from keras.optimizers import SGD


# In[2]:

train_df = pd.read_csv("train_df_after_processing.csv")


# In[3]:

y_train = np.array(train_df['price_doc'])/100000
x_train = np.array(train_df.drop(['id','price_doc'],axis=1))


# In[ ]:

dropout = 0.2
weights = initializers.RandomUniform(minval=-100, maxval=100, seed=2)
bias = 'random_uniform'
res_model_df = pd.DataFrame(columns=['n','mse','val_mse','time_used'])
res_test_dict = {}

for n in (500*2**n for n in range(1,8)):
    model = Sequential()
    model.add(Dense(n, activation='sigmoid', kernel_initializer=weights, bias_initializer=bias, input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout))
    model.add(Dense(n, activation='sigmoid', kernel_initializer=weights, bias_initializer=bias))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    save_best_weights = '2_Layers_'+str(n)+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'_Sberbank_regression_weights.h5'

    t0 = time.time()
    callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')]
    # callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True)]
    history = model.fit(x_train,
                        y_train,
                        batch_size=256,
                        epochs=300, #Use 100, I reduce it for Kaggle,
                        validation_split=0.15,
                        verbose=True,
                        shuffle=True,
                        callbacks=callbacks)
    time_used = round((time.time()-t0)/60,2)
    print("Minutes elapsed: %f" % time_used)
    
    # find best loss and record it
    min_val_loss, idx = min((loss, idx) for (idx, loss) in enumerate(history.history['val_loss']))
    res_model_df.loc[len(res_model_df)] = [n,min_val_loss,history.history['loss'][idx],time_used]

    # Make predictions with the best weights
    model.load_weights(save_best_weights)
    predictions = model.predict(x_train[:,:], verbose = True)
    predictions = predictions.flatten()
    percentage = []
    for i in range(len(predictions)):
        percentage.append((predictions[i]-y_train[i])/y_train[i]*100)
    result = np.array([predictions, y_train[:], percentage]).T
    result_df = pd.DataFrame(result,columns = ['prediction','real value', 'error percentage'])
    result_df.to_csv('2_Layers_'+str(n)+'_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")+'_result.csv')
    np.set_printoptions(precision=1)
    print(result)
    
    res_test_dict[str(n)] = result_df


# In[ ]:

res_dict = {}
res_dict['res_model_df'] = res_model_df
res_dict['res_test_dict'] = res_test_dict
with open('2_Layers_Batch_Train_Test_Result.pickle', 'wb') as handle:
    pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

