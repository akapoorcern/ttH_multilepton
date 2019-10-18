import matplotlib.pyplot as plt
import numpy as np
import numpy
import pandas
import pandas as pd
import optparse, json, argparse
import ROOT
import sys
from array import array
# python looks here for its packages: $PYTHONPATH. Need to add path to $PYTHONPATH so python can find the required packages.
sys.path.insert(0, '/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/keras-DNN/')
from plotting.plotter import plotter
from ROOT import TFile, TTree, gDirectory
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, LabelEncoder
from sklearn.utils import class_weight
import os
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam, Adadelta, Adagrad
from keras.layers import Dropout, Conv1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from root_numpy import root2array, tree2array
#seed = 7
#np.random.seed(7)
#rng = np.random.RandomState(31337)

class apply_DNN(object):

    def __init__(self):
        #self.output_ttree = ''
        pass

    def getEOSlslist(self, directory, mask='', prepend='root://eosuser.cern.ch/'):
        from subprocess import Popen, PIPE

        eos_cmd = '/afs/cern.ch/project/eos/installation/0.3.15/bin/eos.select'
        eos_dir = '/eos/user/%s ' % (directory)
        data = Popen([eos_cmd, prepend, ' ls ', eos_dir], stdout=PIPE)
        out,err = data.communicate()

        full_list = []

        ## if input file was single root file:
        if directory.endswith('.root'):
            if len(out.split('\n')[0]) > 0:
                return [os.path.join(prepend,eos_dir).replace(" ","")]

        ## instead of only the file name append the string to open the file in ROOT
        for line in out.split('\n'):
            if len(line.split()) == 0: continue
            full_list.append(os.path.join(prepend,eos_dir,line).replace(" ",""))

        ## strip the list of files if required
        if mask != '':
            stripped_list = [x for x in full_list if mask in x]
            return stripped_list

        ## return
        return full_list

    def load_data(self, inputPath,variables,criteria,key,fileName):
        print '<apply_DNN> Loading variables to .csv: ', variables
        print '<apply_DNN> Selection: ', criteria
        print '<apply_DNN> key: ', key
        print '<apply_DNN> fileName: ', fileName
        data = pd.DataFrame(columns=variables)

        inputlist = self.getEOSlslist(directory=inputPath+fileName+".root")
        current_file = str(inputlist[0])
        print '<apply_DNN> current_file: ', current_file

        try: tfile = ROOT.TFile(current_file)
        except :
            print " file "+ current_file+" doesn't exist "
            exit(0)
        try: nominal_tree = gDirectory.Get( 'syncTree' )
        except :
            print "Tree doesn't exist in " + current_file
            exit(0)
        if nominal_tree is not None :
            try: chunk_arr = tree2array(tree=nominal_tree, selection=criteria) #,  start=start, stop = stop)
            except :
                print 'Couldnt convert to array'
                exit(0)
            else :
                chunk_df = pd.DataFrame(chunk_arr, columns=variables)
                data=data.append(chunk_df, ignore_index=True)
        else:
            print 'nominal_tree none existent'
        tfile.Close()
        return data

    def baseline_model(self,num_variables,optimizer,nClasses):
        model = Sequential()
        # input_dim = number of variables
        model.add(Dense(32,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
        for index in xrange(5):
            model.add(Dense(16,activation='relu'))
        for index in xrange(5):
            model.add(Dense(16,activation='relu'))
        for index in xrange(5):
            model.add(Dense(8,activation='relu'))
        model.add(Dense(nClasses, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
        return model

    def baseline_CNN_model(self, num_variables, optimizer):
        # Define sequential model
        model = Sequential()
        # Get shape of input dataset
        train_x_shape_dim2 = num_variables
        print 'Conv1D input_shape = (%s, %s)' % (train_x_shape_dim2,1)
        # Create desired network architecture
        model.add( Conv1D( filters=64, kernel_size=4, activation='relu', input_shape=(train_x_shape_dim2,1) ) )
        model.add( Conv1D(64, 4, activation='relu') )
        # Dropout layer to reduce chance of over-training.
        model.add(Dropout(0.5))
        # reduce dimensionality using Max pooling (characterisic maximum value for each patch)
        model.add(MaxPooling1D(pool_size=2))
        # Flattening creates connection between the convolution and dense layers.
        model.add(Flatten())
        # Combine local features of previous convolutional layers.
        model.add(Dense(64,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
        return model

    def load_trained_model(self, weights_path, num_variables, optimizer,nClasses):
        print '<apply_DNN> loading weights_path: ', weights_path
        model = self.baseline_model(num_variables, optimizer,nClasses)
        model.load_weights(weights_path)
        from keras.models import load_model
        return model

    def load_trained_CNN_model(self, weights_path, num_variables, optimizer):
        print 'loading weights_path: ', weights_path
        model = self.baseline_CNN_model(num_variables, optimizer)
        model.load_weights(weights_path)
        from keras.models import load_model
        return model

    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def evaluate_model(self, eventnum_resultsprob_dict, Eventnum_):
        ttH_node_value = -1.
        ttJ_node_value = -1.
        ttW_node_value = -1.
        ttZ_node_value = -1.
        tHQ_node_value = -1. 
        ttH_node_value = eventnum_resultsprob_dict.get(Eventnum_)[0]
        ttJ_node_value = eventnum_resultsprob_dict.get(Eventnum_)[1]
        ttW_node_value = eventnum_resultsprob_dict.get(Eventnum_)[2]
        ttZ_node_value = eventnum_resultsprob_dict.get(Eventnum_)[3]
        tHQ_node_value = eventnum_resultsprob_dict.get(Eventnum_)[4]

        node_values = [ttH_node_value, ttJ_node_value, ttW_node_value, ttZ_node_value, tHQ_node_value]
        return node_values

    def event_categorised_max_value(self, event_classification, evaluated_node_values):

        event_node_values_max_only = [-2,-2,-2,-2,-2]
        #event_node_values_max_only = [-2,-2,-2,-2]
        if event_classification == 0:
            event_node_values_max_only[0] = evaluated_node_values[0]
        elif event_classification == 1:
            event_node_values_max_only[1] = evaluated_node_values[1]
        elif event_classification == 2:
            event_node_values_max_only[2] = evaluated_node_values[2]
        elif event_classification == 3:
            event_node_values_max_only[3] = evaluated_node_values[3]
        elif event_classification == 4:
            event_node_values_max_only[4] = evaluated_node_values[4]

        return event_node_values_max_only
