import matplotlib.pyplot as plt
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



def createmodel(NN,InputN):
    model = Sequential()
        
    if NN==1:
        model.add(Dense(36, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
        model.add(Reshape((6, 6, 1), input_shape = (36, )))
        model.add(Conv2D(72, kernel_size = (3, 3), kernel_initializer = 'glorot_normal',activation ='relu', padding = 'same'))
        model.add(Flatten())
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(2, activation = 'softmax'))
        
    if NN==2:
    
        model.add(Dense(32, kernel_initializer='glorot_normal', activation='relu', input_dim=InputN))
        model.add(Dropout(0.1))
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

    return model
