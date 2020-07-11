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

print("TensorFlow version: ", tf.__version__)
# assert version.parse(tf.__version__).release[0] >= 2, \
#     "This notebook requires TensorFlow 2.0 or above."

import argparse
import sys


# In[2]:

BS=int(sys.argv[1])
LR=float(sys.argv[2])
EP=int(sys.argv[3])
LV=int(sys.argv[4])
NN=int(sys.argv[5])
GPU=int(sys.argv[6])

chain = ROOT.TChain('syncTree')
chain.Add('TTH_ctcvcp_DiLepRegion_New2016.root')
chain.Add('TTH_ctcvcp_DiLepRegion_New2017.root')
chain.Add('TTH_ctcvcp_DiLepRegion_New2018.root')

# In[3]:

treeNP = tree2array(chain)


# In[4]:


odf = pd.DataFrame(treeNP)
#df.to_csv(r'df.csv', index = False)
df = odf.loc[(odf['n_presel_jet']>2.0) & (odf['is_tH_like_and_not_ttH_like'] !=1.0) & (odf['HiggsDecay'] !=1.0)]
#df.to_csv(r'df.csv', index = False)


# In[3]:


listVar1=['mvaOutput_2lss_ttV',
          'avg_dr_jet',
          'dr_leps',
         'massL',
         'deta_highest2b',
         'dphi_highest2b',
         'mindr_lep1_jet',
         'mindr_lep2_jet',
         'lep1_pt',
         'lep1_eta',
         'lep1_phi',
         'lep2_pt',
         'lep2_eta',
         'lep2_phi',
         'jet1_pt',
         'jet2_pt',
         'jet3_pt',
         'jethb1_pt',
         'jethb1_eta',
         'jethb1_E',
         'jethb1_phi',
         'jethb2_pt',
         'jethb2_eta',
         'jethb2_E',
         'jethb2_phi',
         'EVENT_rWeights',
         'EVENT_originalXWGTUP']


listVar2=['angle_bbpp_truth2l2b',
         'cosa_bbpp_truth2l2b',
         'truth_H_eta',
         'truth_H_pt',
         'truth_cosa_2b',
         'truth_deta_2b',
         'truth_hadTop_eta',
         'truth_hadTop_pt',
         'EVENT_rWeights',
         'EVENT_originalXWGTUP']


listVar3=['mvaOutput_2lss_ttV',
         'avg_dr_jet',
         'dr_leps',
         'massL',
         'deta_highest2b',
         'dphi_highest2b',
         'mindr_lep1_jet',
         'mindr_lep2_jet',
         'lep1_pt',
         'lep2_pt',
         'EVENT_rWeights',
         'EVENT_originalXWGTUP']

listVar4=['mvaOutput_2lss_ttV',
          'avg_dr_jet',
          'dr_leps',
          'massL',
          'deta_highest2b',
          'dphi_highest2b',
          'mindr_lep1_jet',
          'mindr_lep2_jet',
          'lep1_pt',
          'lep1_eta',
          'lep1_phi',
          'lep2_pt',
          'lep2_eta',
          'lep2_phi',
          'jet1_pt',
          'jet2_pt',
          'jet3_pt',
          'jethb1_pt',
          'jethb1_eta',
          'jethb1_E',
          'jethb1_phi',
          'jethb2_pt',
          'jethb2_eta',
          'jethb2_E',
          'jethb2_phi',
          #'acuteangle_bbpp_highest2b',
          'angle_bbpp_highest2b',
          #'angle_bbpp_loose2b',
          #'cosa_highest2b',
          'EVENT_rWeights',
          'EVENT_originalXWGTUP']



listVar5=['avg_dr_jet',
          'dr_leps',
          'massL',
          'deta_highest2b',
          'dphi_highest2b',
          'mindr_lep1_jet',
          'mindr_lep2_jet',
          'lep1_pt',
          'lep1_eta',
          'lep1_phi',
          'lep1_E',
          'lep2_pt',
          'lep2_eta',
          'lep2_phi',
          'lep2_E',
         'jethb1_pt',
         'jethb1_eta',
         'jethb1_E',
         'jethb1_phi',
         'jethb2_pt',
         'jethb2_eta',
         'jethb2_E',
         'jethb2_phi',
         'EVENT_rWeights',
         'EVENT_originalXWGTUP']


listVar6=['lep1_pt',
          'lep1_eta',
          'lep1_phi',
          'lep1_E',
          'lep2_pt',
          'lep2_eta',
          'lep2_phi',
          'lep2_E',
         'jet1_pt',
         'jet1_eta',
         'jet1_E',
         'jet1_phi',
         'jet2_pt',
         'jet2_eta',
         'jet2_E',
         'jet2_phi',
         'jet3_pt',
         'jet3_eta',
         'jet3_E',
         'jet3_phi',
         'jet4_pt',
         'jet4_eta',
         'jet4_E',
         'jet4_phi',
         'EVENT_rWeights',
         'EVENT_originalXWGTUP']


listVar7=['mvaOutput_2lss_ttV',
          'avg_dr_jet',
          'dr_leps',
          'massL',
          'deta_highest2b',
          'dphi_highest2b',
          'mindr_lep1_jet',
          'mindr_lep2_jet',
          'lep1_pt',
          'lep1_eta',
          'lep1_phi',
          'lep1_E',
          'lep2_pt',
          'lep2_eta',
          'lep2_phi',
          'lep2_E',
          'jet1_pt',
          'jet1_eta',
          'jet1_E',
          'jet1_phi',
          'jet2_pt',
          'jet2_eta',
          'jet2_E',
          'jet2_phi',
          'jet3_pt',
          'jet3_eta',
          'jet3_E',
          'jet3_phi',
          'jet4_pt',
          'jet4_eta',
          'jet4_E',
          'jet4_phi',
          'EVENT_rWeights',
          'EVENT_originalXWGTUP']

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


# In[7]:


df_odd=df[listVar]
#print(df_odd['EVENT_rWeights'])
df_odd['CPWeighto'] = [x[59] for x in df_odd['EVENT_rWeights']]
df_odd['CPWeightp'] = df_odd['CPWeighto']/df_odd['EVENT_originalXWGTUP']

#df_odd['CPWeight'] = 1
#df_odd['CPWeighto'] = 1
sumwp=sum(df_odd['CPWeightp'])
print(f'CPWeightp sump for odd : {sumwp}')
sumw=5
df_odd['CPWeight']=df_odd['CPWeightp'].mul(sumw)
#print(df_odd['CPWeight'])
df_odd['Output'] = 1
df_odd = df_odd.drop(['EVENT_rWeights'],axis=1)
df_newodd = df_odd.rename(columns={'CPWeight':'Weight','Output':'Category'})
#df_newodd.to_csv(r'Cat1.csv', index = False)
#print(df_odd.iloc[0])

df_even=df[listVar]
#print(df_even['EVENT_rWeights'])

df_even['CPWeighto'] = [x[11] for x in df_even['EVENT_rWeights']]
df_even['CPWeightp'] = df_even['CPWeighto']/df_even['EVENT_originalXWGTUP']


#sumW=sum(df_even['CPWeightp'])
sumWp=sum(df_even['CPWeightp'])
print(f'CPWeightp sump for even : {sumWp}')
sumW=1
df_even['CPWeight']=df_even['CPWeightp'].mul(sumW)
df_even['Output'] = 0
df_even = df_even.drop(['EVENT_rWeights'],axis=1)
df_neweven = df_even.rename(columns={'CPWeight':'Weight','Output':'Category'})
#df_neweven.to_csv(r'Cat0.csv', index = False)
#print(df_even.iloc[0])


df_neweven_forcorr=df_neweven.drop(['Weight','Category','EVENT_originalXWGTUP','CPWeighto','CPWeightp'],axis=1)
df_newodd_forcorr=df_newodd.drop(['Weight','Category','EVENT_originalXWGTUP','CPWeighto','CPWeightp'],axis=1)

evencorr=df_neweven_forcorr.corr()
oddcorr=df_newodd_forcorr.corr()

data=pd.concat([df_neweven,df_newodd])
data = data.sample(frac=1).reset_index(drop=True)

features = data.drop(['Weight','Category','EVENT_originalXWGTUP','CPWeighto','CPWeightp'],axis=1)
Weights = data['Weight']
labels = data['Category']

# featuresodd = df_newodd.drop(['Weight','Category','EVENT_originalXWGTUP','CPWeighto'],axis=1)
# featuresodd = scaler.fit_transform(featuresodd)
# Wt_featuresodd = df_newodd['Weight']

# featureseven = df_neweven.drop(['Weight','Category','EVENT_originalXWGTUP','CPWeighto'],axis=1)
# featureseven = scaler.fit_transform(featureseven)
# Wt_featureseven = df_neweven['Weight']

y=np.ravel(labels)
Wt=np.ravel(Weights)
X=features

X_train, X_test, y_trainN, y_testN, Wt_train, Wt_test = train_test_split(X, y, Wt, test_size=0.2, random_state=42)
#X_train, X_test, y_trainN, y_testN = train_test_split(X, y, test_size=0.33, random_state=42)

#X_train = scaler.fit_transform(X_train)

#X_test = scaler.fit_transform(X_test)

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


# if NN==13:
    
#     model.add(Dense(48, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
#     model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu'))
#     model.add(Dense(8, kernel_initializer='glorot_normal', activation='relu'))
#     model.add(Dropout(0.1))
#     model.add(Dense(2, kernel_initializer='glorot_uniform', activation='softmax'))


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

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']


y_pred_test = model.predict(X_test)
y_pred_test_odd =[]
y_pred_test_even =[]
Wt_test_odd = []
Wt_test_even = []

for i,j,k in zip(y_pred_test,y_test,Wt_test):
    if j[0]==1:
        y_pred_test_odd.append(i[0])
        Wt_test_odd.append(k)
    if j[0]==0:
        y_pred_test_even.append(i[0])
        Wt_test_even.append(k)

y_pred_train = model.predict(X_train)
y_pred_train_odd =[]
y_pred_train_even =[]
Wt_train_odd = []
Wt_train_even = []

for i,j,k in zip(y_pred_train,y_train,Wt_train):
    if j[0]==1:
        y_pred_train_odd.append(i[0])
        Wt_train_odd.append(k)
    if j[0]==0:
        y_pred_train_even.append(i[0])
        Wt_train_even.append(k)

y_test_o=[]
y_pred_test_o=[]

for item in y_test:
    y_test_o.append(item[0])
for item in y_pred_test:
    y_pred_test_o.append(item[0])

fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_o, y_pred_test_o, sample_weight=Wt_test)
auc_keras = roc_auc_score(y_test_o, y_pred_test_o, sample_weight=Wt_test)


# In[9]:



from mpl_toolkits.axes_grid.inset_locator import inset_axes

fig, axes = plt.subplots(2,3,figsize=(20,10))
ax1=axes[0,0]
ax2=axes[0,1]
ax3=axes[0,2]
ax4=axes[1,0]
ax5=axes[1,1]
ax6=axes[1,2]

ax1.plot(loss, color='y')
ax1.plot(val_loss, color='c')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
#meter that I can replace in the above code that will allow custom locations of the inset axes within the parent axes? I've tried to use the bbox_to_ancax1.set_yscale('log')
#ax1.set_ylim(0.55, 0.8)
ax1.legend(['loss', 'val_loss'], loc=2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

loss_20 = loss[-20:]
val_loss_20 = val_loss[-20:]


inset_axes = inset_axes(ax1,
                    width="40%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc=1)


inset_axes.plot(loss_20, color='y')
inset_axes.plot(val_loss_20, color='c')
inset_axes.set_xlabel('Last 20 Epochs')
inset_axes.set_ylabel('Loss')
inset_axes.spines['top'].set_visible(False)
inset_axes.spines['right'].set_visible(False)
inset_axes.spines['bottom'].set_visible(False)
inset_axes.get_xaxis().set_ticks([])
#inset_axes.spines['right'].set_visible(False)
#inset_axes.set_yscale('log')
#ax1.set_ylim(0.55, 0.8)
#inset_axes.legend(['loss', 'val_loss'])

ax2.hist(y_pred_test_even, 25, density=1, histtype='step', weights=Wt_test_even, color='r', label='Test_even')
ax2.hist(y_pred_test_odd, 25, density=1, histtype='step', weights=Wt_test_odd, color='g', label='Test_odd')
ax2.hist(y_pred_train_even, 25, density=1, histtype='bar', weights=Wt_train_even, color='r', label='Train_even', alpha=0.3)
ax2.hist(y_pred_train_odd, 25, density=1, histtype='bar', weights=Wt_train_odd, color='g', label='Train_odd', alpha=0.3)
ax2.set_xlabel('Output')
ax2.set_ylabel('Events')
ax2.set_yscale('log')
ax2.legend(loc=4)
#ax2.set_title('Histogram of Output')
#ax2.text(60, .025, r'$\mu=100,\ \sigma=15$')
ax2.set_xlim(0, 1)
#ax2.ylim(0, 0.03)
ax2.grid(True)


ax4.hist(y_pred_test_even, 25, density=1, histtype='step', weights=Wt_test_even, color='r', label='Test_even')
ax4.hist(y_pred_test_odd, 25, density=1, histtype='step', weights=Wt_test_odd, color='g', label='Test_odd')
ax4.hist(y_pred_train_even, 25, density=1, histtype='bar', weights=Wt_train_even, color='r', label='Train_even', alpha=0.3)
ax4.hist(y_pred_train_odd, 25, density=1, histtype='bar', weights=Wt_train_odd, color='g', label='Train_odd', alpha=0.3)
ax4.set_xlabel('Output')
ax4.set_ylabel('Events')
#ax4.set_yscale('log')
ax4.legend(loc=4)
#ax4.set_title('Histogram of Output')
#ax4.text(60, .025, r'$\mu=100,\ \sigma=15$')
ax4.set_xlim(0, 1)
#ax4.ylim(0, 0.03)
ax4.grid(True)


ax3.plot([0, 1], [0, 1], 'k--')
ax3.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
ax3.set_xlabel('False positive rate')
ax3.set_ylabel('True positive rate')
ax3.set_title('ROC curve')
ax3.legend(loc=4)



sns.heatmap(evencorr, xticklabels=evencorr.columns, yticklabels=evencorr.columns, cmap='RdYlGn', center=0, annot=False,ax=ax5)
# Decorations
ax5.set_title('Even', fontsize=20)
#ax5.set_xticks(fontsize=4)
#ax5.set_yticks(fontsize=4)


sns.heatmap(oddcorr, xticklabels=oddcorr.columns, yticklabels=oddcorr.columns, cmap='RdYlGn', center=0, annot=False,ax=ax6)
# Decorations
ax6.set_title('Odd', fontsize=20)
#ax6.set_xticks(fontsize=4)
#ax6.set_yticks(fontsize=4)

#plt.legend(loc='best',prop={'size': 6})
#plt.legend(loc=4)
plt.tight_layout()
plt.savefig(f'results/foo_withAdj_v1_BS{BS}_EP{EP}_LR{LR}_LV{LV}_NN{NN}_GPU{GPU}.png',dpi=300)
plt.savefig(f'results/foo_withAdj_v1_BS{BS}_EP{EP}_LR{LR}_LV{LV}_NN{NN}_GPU{GPU}.pdf')


# In[ ]:




