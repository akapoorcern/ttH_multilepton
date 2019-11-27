import matplotlib.pyplot as plt
import numpy as np
import numpy
import pandas
import pandas as pd
import optparse, json, argparse
import ROOT
import sys
from array import array
import commands
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
os.environ['EOS_MGM_URL'] = 'root://eosuser.cern.ch/'
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

class apply_DNN(object):

    def __init__(self):
        #self.output_ttree = ''
        pass

    def getEOSlslist(self, directory, mask='', prepend='root://eosuser.cern.ch'):
        eos_dir = '/eos/user/%s ' % (directory)
        eos_cmd = 'eos ' + prepend + ' ls ' + eos_dir
        out = commands.getoutput(eos_cmd)
        full_list = []
        ## if input file was single root file:
        if directory.endswith('.root'):
            if len(out.split('\n')[0]) > 0:
                return [os.path.join(prepend,eos_dir).replace(" ","")]
        ## instead of only the file name append the string to open the file in ROOT
        for line in out.split('\n'):
            print 'line: ', line
            if len(line.split()) == 0: continue
            full_list.append(os.path.join(prepend,eos_dir,line).replace(" ",""))
        ## strip the list of files if required
        if mask != '':
            stripped_list = [x for x in full_list if mask in x]
            return stripped_list
        print 'full files list from eos: ', full_list
        ## return
        return full_list

    def load_data(self, inputPath, variables, criteria, key, fileName):
        my_cols_list=variables+['process', 'totalWeight']
        data = pd.DataFrame(columns=my_cols_list)
        eos_root_filename = inputPath+fileName+".root"

        if 'ttH' in eos_root_filename or 'TTH' in eos_root_filename:
            sampleName='ttH'
            target=0
        if 'Fakes_' in eos_root_filename:
            sampleName='Fakes'
            target=1
        if 'Flips_' in eos_root_filename:
            sampleName='Flips'
            target=1
        if 'ttW_' in eos_root_filename or 'TTW_' in eos_root_filename:
            sampleName='ttW'
            target=2
        if 'THQ' in eos_root_filename:
            sampleName='THQ'
            target=3
        if 'Conv' in eos_root_filename:
            sampleName='Conv'
            target=4
        if 'ttZ' in eos_root_filename or 'TTZ_' in eos_root_filename:
            sampleName='ttZ'
            target=4
        if 'TTWW_' in eos_root_filename:
            sampleName='TTWW'
            target=4
        if 'THW' in eos_root_filename:
            sampleName='THW'
            target=4
        if 'EWK' in eos_root_filename:
            sampleName='EWK'
            target=4
        if 'Rares' in eos_root_filename:
            sampleName='Rares'
            target=4
        if 'FakeSub' in eos_root_filename:
            sampleName='FakeSub'
            target=4
        if 'ttbar_closure' in eos_root_filename:
            sampleName='ttbar_closure'
            target=4
        if 'Data' in eos_root_filename:
            sampleName='Data'
            target=4

        inputlist = self.getEOSlslist(directory=eos_root_filename)
        current_file = str(inputlist[0])

        try: tfile = ROOT.TFile(current_file)
        except :
            print " file "+ current_file+" doesn't exist "
            exit(0)
        try: nominal_tree = gDirectory.Get( 'syncTree' )
        except :
            print "Tree doesn't exist in " + current_file
            exit(0)
        if nominal_tree is not None :
            try: chunk_arr = tree2array(tree=nominal_tree, selection=criteria) #,  start=start, stop=stop)
            except :
                print 'Couldnt convert to array'
                exit(0)
            else :
                chunk_df = pd.DataFrame(chunk_arr, columns=variables)
                chunk_df['process']=sampleName
                chunk_df['totalWeight']=chunk_df['EventWeight']
                chunk_df[['jet1_eta','jet2_eta','jet3_eta','jet4_eta','jetFwd1_eta']] = chunk_df[['jet1_eta','jet2_eta','jet3_eta','jet4_eta','jetFwd1_eta']].apply(np.absolute)
                data=data.append(chunk_df, ignore_index=True)
        else:
            print 'nominal_tree none existent'
        tfile.Close()
        print 'sampleName = ', sampleName
        print "TotalWeights = %f" % ( data.ix[(data.process.values==sampleName)]["totalWeight"].sum() )
        return data

    def baseline_model(self,num_variables,optimizer,nClasses):
        model = Sequential()
        # input_dim = number of variables
        model.add(Dense(32,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
        for index in xrange(5):
            model.add(Dense(16,activation='relu'))
            #model.add(Dropout(0.01))
        for index in xrange(5):
            model.add(Dense(16,activation='relu'))
        for index in xrange(5):
            model.add(Dense(8,activation='relu'))
        model.add(Dense(nClasses, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
        return model

    def load_trained_model(self, weights_path, num_variables, optimizer, nClasses):
        print '<apply_DNN> loading weights_path: ', weights_path
        print '<apply_DNN> loading num_variables: ', num_variables
        print '<apply_DNN> loading optimizer: ', optimizer
        print '<apply_DNN> loading nClasses: ', nClasses
        model = self.baseline_model(num_variables, optimizer, nClasses)
        model.load_weights(weights_path)
        return model

    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def evaluate_model(self, eventnum_resultsprob_dict, Eventnum_):
        ttH_node_value = -1.
        Other_node_value = -1.
        ttW_node_value = -1.
        tHQ_node_value = -1.
        ttH_node_value = eventnum_resultsprob_dict.get(Eventnum_)[0]
        Other_node_value = eventnum_resultsprob_dict.get(Eventnum_)[1]
        ttW_node_value = eventnum_resultsprob_dict.get(Eventnum_)[2]
        tHQ_node_value = eventnum_resultsprob_dict.get(Eventnum_)[3]

        node_values = [ttH_node_value, Other_node_value, ttW_node_value, tHQ_node_value]
        return node_values

    def event_categorised_max_value(self, event_classification, evaluated_node_values):

        event_node_values_max_only = [-2,-2,-2,-2]
        if event_classification == 0:
            event_node_values_max_only[0] = evaluated_node_values[0]
        elif event_classification == 1:
            event_node_values_max_only[1] = evaluated_node_values[1]
        elif event_classification == 2:
            event_node_values_max_only[2] = evaluated_node_values[2]
        elif event_classification == 3:
            event_node_values_max_only[3] = evaluated_node_values[3]

        return event_node_values_max_only
