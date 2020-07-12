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
from mpl_toolkits.axes_grid.inset_locator import inset_axes


def plotit(train_history, model,X_train,y_train,Wt_train,X_test,y_test,Wt_test,arg,evencorr,oddcorr):
    # global inset_axes
    # global roc_curve
    # global roc_auc_score
    
    # X_train=Xdata[0]
    # y_train=ydata[0]
    # Wt_train=Wts[0]
    
    # X_test=Xdata[1]
    # y_test=ydata[1]
    # Wt_test=Wts[1]
    
    BS=int(arg[1])
    LR=float(arg[2])
    EP=int(arg[3])
    LV=int(arg[4])
    NN=int(arg[5])
    GPU=int(arg[6])

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
    
    
    inset_axes1 = inset_axes(ax1,
                            width="40%", # width = 30% of parent_bbox
                            height="40%", # height : 1 inch
                            loc=1)

    inset_axes1.plot(loss_20, color='y')
    inset_axes1.plot(val_loss_20, color='c')
    inset_axes1.set_xlabel('Last 20 Epochs')
    inset_axes1.set_ylabel('Loss')
    inset_axes1.spines['top'].set_visible(False)
    inset_axes1.spines['right'].set_visible(False)
    inset_axes1.spines['bottom'].set_visible(False)
    inset_axes1.get_xaxis().set_ticks([])

    ax2.hist(y_pred_test_even, 25, density=1, histtype='step', weights=Wt_test_even, color='r', label='Test_even')
    ax2.hist(y_pred_test_odd, 25, density=1, histtype='step', weights=Wt_test_odd, color='g', label='Test_odd')
    ax2.hist(y_pred_train_even, 25, density=1, histtype='bar', weights=Wt_train_even, color='r', label='Train_even', alpha=0.3)
    ax2.hist(y_pred_train_odd, 25, density=1, histtype='bar', weights=Wt_train_odd, color='g', label='Train_odd', alpha=0.3)
    ax2.set_xlabel('Output')
    ax2.set_ylabel('Events')
    ax2.set_yscale('log')
    ax2.legend(loc=4)
    ax2.set_xlim(0, 1)
    ax2.grid(True)
    
    ax4.hist(y_pred_test_even, 25, density=1, histtype='step', weights=Wt_test_even, color='r', label='Test_even')
    ax4.hist(y_pred_test_odd, 25, density=1, histtype='step', weights=Wt_test_odd, color='g', label='Test_odd')
    ax4.hist(y_pred_train_even, 25, density=1, histtype='bar', weights=Wt_train_even, color='r', label='Train_even', alpha=0.3)
    ax4.hist(y_pred_train_odd, 25, density=1, histtype='bar', weights=Wt_train_odd, color='g', label='Train_odd', alpha=0.3)
    ax4.set_xlabel('Output')
    ax4.set_ylabel('Events')
    ax4.legend(loc=4)
    ax4.set_xlim(0, 1)
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
    sns.heatmap(oddcorr, xticklabels=oddcorr.columns, yticklabels=oddcorr.columns, cmap='RdYlGn', center=0, annot=False,ax=ax6)
    # Decorations
    ax6.set_title('Odd', fontsize=20)

    plt.tight_layout()
    plt.savefig(f'results/foo_withAdj_v1_BS{BS}_EP{EP}_LR{LR}_LV{LV}_NN{NN}_GPU{GPU}.png',dpi=300)
    plt.savefig(f'results/foo_withAdj_v1_BS{BS}_EP{EP}_LR{LR}_LV{LV}_NN{NN}_GPU{GPU}.pdf')

def CPdataset(df,index,weight,listVar,Cat):

    df=df[listVar]
    df['CPWeighto'] = [x[index] for x in df['EVENT_rWeights']]
    df['CPWeightp'] = df['CPWeighto']/df['EVENT_originalXWGTUP']
    sumWp=sum(df['CPWeightp'])
    print(f'CPWeightp sump for index {index} : {sumWp}')
    sumW=weight
    df['CPWeight']=df['CPWeightp'].mul(sumW)
    df['Output'] = Cat
    df = df.drop(['EVENT_rWeights'],axis=1)
    df_new= df.rename(columns={'CPWeight':'Weight','Output':'Category'})
    df_new_forcorr=df_new.drop(['Weight','Category','EVENT_originalXWGTUP','CPWeighto','CPWeightp'],axis=1)
    return df_new, df_new_forcorr
