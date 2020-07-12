from numpy.random import seed
seed(1)
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
from DNNs import *

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
tf.random.set_seed(2) 
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
#####################################################################

def SingleModelResult(InputN, arg, Xdata, Ydata, Wts, evencorr, oddcorr, tensorboard_callback):

    BS=int(arg[1])
    LR=float(arg[2])
    EP=int(arg[3])
    LV=int(arg[4])
    NN=int(arg[5])
    GPU=int(arg[6])

    model=createmodel(NN,InputN)

    if GPU>1:
        model = multi_gpu_model(model, gpus=GPU)

    print("[INFO] compiling model...")
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LR), metrics=['accuracy',])
    model.summary()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    train_history = model.fit(Xdata[0],ydata[0],
                              epochs=EP,batch_size=BS,
                              validation_data=(Xdata[1],ydata[1],Wts[1]),verbose=1,
                              sample_weight=Wts[0],
                              callbacks=[es, tensorboard_callback])
    plotit(train_history, model,Xdata[0],ydata[0],Wts[0],Xdata[1],ydata[1],Wts[1],arg,evencorr,oddcorr)
    return train_history

print("TensorFlow version: ", tf.__version__)

#####################################################################
#####Start Code

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

Xdata=[X_train,X_test]
ydata=[y_train,y_test]
Wts=[Wt_train,Wt_test]

train_history=SingleModelResult(InputN, sys.argv, Xdata, ydata, Wts, evencorr, oddcorr, tensorboard_callback)
