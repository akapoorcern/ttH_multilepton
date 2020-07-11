import os
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import optparse, json, argparse, math
import ROOT
from helpers import *
from Varlist import *

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from tensorflow.compat.v2.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from root_numpy import root2array, tree2array
import argparse
import sys

print("TensorFlow version: ", tf.__version__)

BS=int(sys.argv[1])
LR=float(sys.argv[2])
EP=int(sys.argv[3])
LV=int(sys.argv[4])
NN=int(sys.argv[5])
GPU=int(sys.argv[6])

chain = ROOT.TChain('syncTree')
chain.Add('samples/TTH_ctcvcp_DiLepRegion_New2016.root')
chain.Add('samples/TTH_ctcvcp_DiLepRegion_New2017.root')
chain.Add('samples/TTH_ctcvcp_DiLepRegion_New2018.root')

treeNP = tree2array(chain)

odf = pd.DataFrame(treeNP)
df = odf.loc[(odf['n_presel_jet']>2.0) & (odf['is_tH_like_and_not_ttH_like'] !=1.0) & (odf['HiggsDecay'] !=1.0)]

listVar=[]
if LV==1:
    listVar=listVar1
if LV==2:
    listVar=listVar2
if LV==3:
    listVar=listVar3
if LV==4:
    listVar=listVar4
if LV==5:
    listVar=listVar5
if LV==6:
    listVar=listVar6
if LV==7:
    listVar=listVar7

InputN=len(listVar)-2


df_neweven, df_neweven_forcorr= CPdataset(df,11,1,listVar,0)
df_newodd, df_newodd_forcorr= CPdataset(df,59,5,listVar,1)

# df_even=df[listVar]
# df_even['CPWeighto'] = [x[11] for x in df_even['EVENT_rWeights']]
# df_even['CPWeightp'] = df_even['CPWeighto']/df_even['EVENT_originalXWGTUP']
# sumWp=sum(df_even['CPWeightp'])
# print(f'CPWeightp sump for even : {sumWp}')
# sumW=1
# df_even['CPWeight']=df_even['CPWeightp'].div(sumW)
# df_even['Output'] = 0
# df_even = df_even.drop(['EVENT_rWeights'],axis=1)
# df_neweven = df_even.rename(columns={'CPWeight':'Weight','Output':'Category'})
# df_neweven_forcorr=df_neweven.drop(['Weight','Category','EVENT_originalXWGTUP','CPWeighto','CPWeightp'],axis=1)


# df_odd=df[listVar]
# #print(df_odd['EVENT_rWeights'])
# df_odd['CPWeighto'] = [x[59] for x in df_odd['EVENT_rWeights']]
# df_odd['CPWeightp'] = df_odd['CPWeighto']/df_odd['EVENT_originalXWGTUP']
# #df_odd['CPWeight'] = 1
# #df_odd['CPWeighto'] = 1
# sumwp=sum(df_odd['CPWeightp'])
# print(f'CPWeightp sump for odd : {sumwp}')
# sumw=5
# df_odd['CPWeight']=df_odd['CPWeightp'].mul(sumw)
# #print(df_odd['CPWeight'])
# df_odd['Output'] = 1
# df_odd = df_odd.drop(['EVENT_rWeights'],axis=1)
# df_newodd = df_odd.rename(columns={'CPWeight':'Weight','Output':'Category'})
# #df_newodd.to_csv(r'Cat1.csv', index = False)
# #print(df_odd.iloc[0])
# df_newodd_forcorr=df_newodd.drop(['Weight','Category','EVENT_originalXWGTUP','CPWeighto','CPWeightp'],axis=1)

evencorr=df_neweven_forcorr.corr()
oddcorr=df_newodd_forcorr.corr()

data=pd.concat([df_neweven,df_newodd])
data = data.sample(frac=1).reset_index(drop=True)

features = data.drop(['Weight','Category','EVENT_originalXWGTUP','CPWeighto','CPWeightp'],axis=1)
Weights = data['Weight']
labels = data['Category']

y=np.ravel(labels)
Wt=np.ravel(Weights)
X=features

X_train, X_test, y_trainN, y_testN, Wt_train, Wt_test = train_test_split(X, y, Wt, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform (X_test)

y_train= to_categorical(y_trainN, num_classes=2)
y_test= to_categorical(y_testN, num_classes=2)


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


tensorboard_callback = TensorBoard(log_dir=f'LOGGY2/logdir_BS{BS}_EP{EP}_LR{LR}_LV{LV}_NN{NN}_GPU{GPU}_withAdj_v1')

tf.compat.v1.disable_eager_execution()
print("[INFO] training with {} GPUs...".format(GPU))

model = Sequential()

if NN==1:
    model.add(Dense(36, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
    model.add(Reshape((6, 6, 1), input_shape = (36, )))
    model.add(Conv2D(72, kernel_size = (3, 3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same'))
    #model.add(BatchNormalization())
    #model.add(Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same'))
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))

if NN==2:
    
    model.add(Dense(32, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
    model.add(Dropout(0.1))
    #model.add(Dense(32, kernel_initializer='glorot_normal', activation='relu'))
    #model.add(Dropout(0.1))
    model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(8, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, kernel_initializer='glorot_uniform', activation='softmax'))

if NN==3:
    
    model.add(Dense(36, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
    model.add(Reshape((6, 6, 1), input_shape = (36, )))
    model.add(Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same'))
    model.add(Flatten())
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))

if NN==4:
    
    model.add(Dense(100, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
    for x in range(1):
        model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='glorot_uniform', activation='softmax'))

if NN==5:
    
    model.add(Dense(36, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
    model.add(Reshape((6, 6, 1), input_shape = (25, )))
    model.add(Conv2D(8, kernel_size = (3, 3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same'))
    #model.add(BatchNormalization())
    model.add(Conv2D(8, kernel_size = (3, 3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same'))
    model.add(Flatten())
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))

if NN==6:
    model.add(Dense(36, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
    model.add(Reshape((6, 6, 1), input_shape = (25, )))
    model.add(Conv2D(16, kernel_size = (3, 3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same'))
    model.add(Conv2D(16, kernel_size = (3, 3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same'))
    model.add(Flatten())
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(2, activation = 'softmax'))
    
if NN==10:
    model.add(Dense(100, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
    model.add(Dense(100, kernel_initializer='glorot_normal', activation='relu'))
    for x in range(1):
        model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
        model.add(Dropout(0.01))
    model.add(Dense(2, kernel_initializer='glorot_normal', activation='softmax'))

if NN==11:
    
    model.add(Dense(32, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
    model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, kernel_initializer='glorot_uniform', activation='softmax'))


if NN==12:
    
    model.add(Dense(48, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
    model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, kernel_initializer='glorot_uniform', activation='softmax'))

if NN==13:
    
    model.add(Dense(100, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
    model.add(Dense(50, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, kernel_initializer='glorot_uniform', activation='softmax'))


if GPU>1:
    model = multi_gpu_model(model, gpus=GPU)

print("[INFO] compiling model...")
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy',])
model.summary()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    
train_history = model.fit(X_train, y_train,epochs=EP, batch_size=BS, validation_data=(X_test, y_test, Wt_test), verbose=1,sample_weight=Wt_train, 
                          callbacks=[es, tensorboard_callback])

plotit(train_history, model,X_train,y_train,Wt_train,X_test,y_test,Wt_test,sys.argv,evencorr,oddcorr)
