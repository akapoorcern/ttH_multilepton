# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           train-DNN.py
#  Author: Joshuha Thomas-Wilsker
#  Institute of High Energy Physics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Code to train deep neural network
# for ttH multilepton analysis.
# Information on regions below used
# to calculate class weights to
# help with the class imbalance in
# the analysis.
# =============== Weights ==================
# WARNING! 'sample_weight' will overide 'class_weight'
# ==========================================

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
import os
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from plotting.plotter import plotter
from root_numpy import root2array, tree2array
seed = 7
np.random.seed(7)
rng = np.random.RandomState(31337)

def load_data(inputPath,variables,criteria):
    # Load dataset to .csv format file
    my_cols_list=variables+['process', 'key', 'target', 'totalWeight','sampleWeight']
    data = pd.DataFrame(columns=my_cols_list)
    #keys=['ttH','ttJ','ttW','THQ']
    keys=['THQ','ttH','ttJ','ttW']

    for key in keys :
        print key
        if 'ttH' in key or 'TTH' in key:
            sampleNames=['ttH']
            fileNames = ['ttHnobb_DiLepRegion']
            target=0
        if 'ttJ' in key:
            sampleNames=['ttJ']
            fileNames = ['ttJets_PS_DiLepRegion']
            target=1
        if 'ttW' in key or 'TTW' in key:
            sampleNames=['ttW']
            fileNames=['ttWJets_DiLepRegion']
            target=2
        if 'THQ' in key:
            sampleNames=['THQ']
            fileNames=['THQ_ctcvcp_DiLepRegion']
            #criteria = "(" + criteria + "&&(nEvent%3==0))"
            target=3
        if 'ttZ' in key:
            sampleNames=['ttZ']
            fileNames = ['ttZJets_DiLepRegion']
            target=4

        inputTree = 'syncTree'

        for process_index in xrange(len(fileNames)):
            fileName = fileNames[process_index]
            sampleName = sampleNames[process_index]

            try: tfile = ROOT.TFile(inputPath+"/"+fileName+".root")
            except :
                print " file "+ inputPath+"/"+fileName+".root doesn't exits "
                continue
            try: tree = tfile.Get(inputTree)
            except :
                print inputTree + " deosn't exists in " + inputPath+"/"+fileName+".root"
                continue
            if tree is not None :
                print 'criteria: ', criteria
                #try: chunk_arr = tree2array(tree=tree, selection=criteria)#, start=0, stop=100) # Can use  start=first entry, stop = final entry desired
                try: chunk_arr = tree2array(tree=tree, selection=criteria, start=0, stop=100) # Can use  start=first entry, stop = final entry desired
                except : continue
                else :
                    chunk_df = pd.DataFrame(chunk_arr, columns=variables)
                    chunk_df['process']=sampleName
                    chunk_df['key']=key
                    chunk_df['target']=target
                    chunk_df['totalWeight']=chunk_df["EventWeight"]
                    chunk_df['nEvent']=chunk_df["nEvent"]

                    '''
                    2017 samples
                    # Tuned weights for when not using event weights:
                    Process ttH frequency:  333821
                    Target ttJ frequency:  517071
                    Process ttW frequency:  328345
                    Process tHQ frequency:  95686 # 148233
                    Target ttZ frequency:  117993
                    '''

                    # old weights: tuned_weighted = {0 : 7.67, 1 : 1.0, 2 : 4.62, 3 : 7.67}
                    if sampleName=='ttH':
                        chunk_df['sampleWeight'] = 1.5489468907
                    if sampleName=='ttJ':
                        chunk_df['sampleWeight'] = 1.0
                    if sampleName=='ttW':
                        chunk_df['sampleWeight'] = 1.57477957636
                    if sampleName=='THQ':
                        chunk_df['sampleWeight'] = 5.40383128148
                    if sampleName=='ttZ':
                        chunk_df['sampleWeight'] = 1.0

                    chunk_df[['jet1_eta','jet2_eta','jet3_eta','jet4_eta','jetFwd1_eta']] = chunk_df[['jet1_eta','jet2_eta','jet3_eta','jet4_eta','jetFwd1_eta']].apply(np.absolute)
                    data = data.append(chunk_df, ignore_index=True)
            tfile.Close()
        if len(data) == 0 : continue
        nttH = len(data.ix[(data.target.values == 0) & (data.key.values==key) ])
        nttJ = len(data.ix[(data.target.values == 1) & (data.key.values==key) ])
        nttW = len(data.ix[(data.target.values == 2) & (data.key.values==key) ])
        nTHQ = len(data.ix[(data.target.values == 3) & (data.key.values==key) ])
        nttZ = len(data.ix[(data.target.values == 4) & (data.key.values==key) ])
        nOther = len(data.ix[(data.target.values == 5) & (data.key.values==key) ])
        processfreq = data.groupby('key')
        samplefreq = data.groupby('process')
        if key == 'ttH':
            print 'Process ttH frequency: ', len(processfreq.get_group('ttH'))
        elif key == 'ttJ':
            print 'Target ttJ frequency: ', len(samplefreq.get_group('ttJ'))
        elif key == 'ttW':
            print 'Process ttW frequency: ', len(processfreq.get_group('ttW'))
        elif key == 'THQ':
            print 'Process tHQ frequency: ', len(processfreq.get_group('THQ'))
        elif key == 'ttZ':
            print 'Target ttZ frequency: ', len(samplefreq.get_group('ttZ'))
        elif key == 'Other':
            print 'Process Other frequency: ', len(processfreq.get_group('Other'))
        print "TotalWeights = %f" % (data.ix[(data.key.values==key)]["totalWeight"].sum())
        nNW = len(data.ix[(data["totalWeight"].values < 0) & (data.key.values==key) ])
        print key, "events with -ve weights", nNW
    print '<load_data> data columns: ', (data.columns.values.tolist())
    n = len(data)
    nttH = len(data.ix[data.target.values == 0])
    nttJ = len(data.ix[data.target.values == 1])
    nttW = len(data.ix[data.target.values == 2])
    nTHQ = len(data.ix[data.target.values == 3])
    nttZ = len(data.ix[data.target.values == 4])
    nOther = len(data.ix[data.target.values == 5])
    print "Total length of nttH = %i, ttJ = %i, nttW = %i, nTHQ = %i" % (nttH, nttJ , nttW, nTHQ)
    return data

def load_trained_model(weights_path, num_variables, optimizer,nClasses):
    model = baseline_model(num_variables, optimizer,nClasses)
    model.load_weights(weights_path)
    return model

def normalise(x_train, x_test):
    mu = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_normalised = (x_train - mu) / std
    x_test_normalised = (x_test - mu) / std
    return x_train_normalised, x_test_normalised

def baseline_model(num_variables,optimizer,nClasses):
    model = Sequential()
    model.add(Dense(32,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    for index in xrange(5):
        model.add(Dense(16,activation='relu'))
        model.add(Dropout(0.01))
    for index in xrange(5):
        model.add(Dense(16,activation='relu'))
    for index in xrange(5):
        model.add(Dense(8,activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))
    if optimizer=='Adam':
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['acc'])
    if optimizer=='Nadam':
        model.compile(loss='categorical_crossentropy',optimizer=Nadam(lr=0.001),metrics=['acc'])
    return model

def check_dir(dir):
    if not os.path.exists(dir):
        print 'mkdir: ', dir
        os.makedirs(dir)

# Ratio always > 1. mu use in natural log multiplied into ratio. Keep mu above 1 to avoid class weights going negative.
def create_class_weight(labels_dict,mu=0.9):
    total = np.sum(labels_dict.values()) # total number of examples in all datasets
    keys = labels_dict.keys() # labels
    class_weight = dict()
    print 'total: ', total

    for key in keys:
        # logarithm smooths the weights for very imbalanced classes.
        score = math.log(mu*total/float(labels_dict[key])) # natlog(parameter * total number of examples / number of examples for given label)
        #score = float(total/labels_dict[key])
        print 'score = ', score
        if score > 0.:
            class_weight[key] = score
        else :
            class_weight[key] = 1.
    return class_weight

def main():
    print 'Using Keras version: ', keras.__version__

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-t', '--train_model', dest='train_model', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=0, type=int)
    parser.add_argument('-w', '--classweights', dest='classweights', help='Option to choose class weights', default='InverseSRYields', type=str)
    parser.add_argument('-s', '--sel', dest='selection', help='Option to choose selection', default='tH', type=str)
    args = parser.parse_args()
    do_model_fit = args.train_model
    classweights_name = args.classweights
    selection = args.selection

    # Number of classes to use
    number_of_classes = 4

    # Create instance of output directory where all results are saved.
    output_directory = '2017samples_%s_%s/' % (selection,classweights_name)

    check_dir(output_directory)

    # Create plots subdirectory
    plots_dir = os.path.join(output_directory,'plots/')

    input_var_jsonFile = open('input_vars_SigRegion_wFwdJet.json','r')

    if selection == 'tH':
        selection_criteria = '(is_tH_like_and_not_ttH_like==0 || is_tH_like_and_not_ttH_like==1)'#&& n_presel_jet>=3'

    # Load Variables from .json
    variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()

    # Create list of headers for dataset .csv
    column_headers = []
    for key,var in variable_list:
        column_headers.append(key)
    column_headers.append('EventWeight')
    column_headers.append('xsec_rwgt')
    column_headers.append('nEvent')

    # Create instance of the input files directory
    inputs_file_path = '/eos/user/j/jthomasw/4anshul/rootplas_LegacyMVA_1113/DiLepRegion/ttH2017TrainDNN2L/'

    # Load ttree into .csv including all variables listed in column_headers
    print '<train-DNN> Input file path: ', inputs_file_path
    outputdataframe_name = '%s/output_dataframe_%s.csv' %(output_directory,selection)
    if os.path.isfile(outputdataframe_name):
        data = pandas.read_csv(outputdataframe_name)
        print '<train-DNN> Loading data .csv from: %s . . . . ' % (outputdataframe_name)
    else:
        print '<train-DNN> Creating new data .csv @: %s . . . . ' % (inputs_file_path)
        data = load_data(inputs_file_path,column_headers,selection_criteria)
        data.to_csv(outputdataframe_name, index=False)
        data = pandas.read_csv(outputdataframe_name)

    # Make instance of plotter tool
    Plotter = plotter()

    # Create statistically independant lists train/test data (used to train/evaluate the network)
    traindataset, valdataset = train_test_split(data, test_size=0.2)
    #valdataset.to_csv('valid_dataset.csv', index=False)

    #print '<train-DNN> Training dataset shape: ', traindataset.shape
    #print '<train-DNN> Validation dataset shape: ', valdataset.shape

    # Remove last two columns (Event weight and xsrw) from column headers
    training_columns = column_headers[:-3]
    print '<train-DNN> Training features: ', training_columns

    # Select data from columns under the remaining column headers in traindataset
    X_train = traindataset[training_columns].values

    # Select data from 'target' as target for MVA
    Y_train = traindataset.target.astype(int)
    X_test = valdataset[training_columns].values
    Y_test = valdataset.target.astype(int)

    num_variables = len(training_columns)

    # Create dataframe containing input features only (for correlation matrix)
    train_df = data.iloc[:traindataset.shape[0]]
    train_df.drop(['EventWeight'], axis=1, inplace=True)
    train_df.drop(['xsec_rwgt'], axis=1, inplace=True)

    ## Input Variable Correlation plot
    correlation_plot_file_name = 'correlation_plot.png'
    #Plotter.correlation_matrix(train_df)
    #Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name)

    #sampleweights = traindataset.loc[:,'sampleWeight']*traindataset.loc[:,'EventWeight']
    sampleweights = traindataset.loc[:,'sampleWeight']
    sampleweights = np.array(sampleweights)

    # Dictionaries of class weights to combat class imbalance
    if classweights_name == 'balanced':
        tuned_weighted = class_weight.compute_class_weight('balanced', np.unique([0,1,2,3]), Y_train)
    if classweights_name == 'tunedweights':
        tuned_weighted = {0 : 7.67, 1 : 1.0, 2 : 4.62, 3 : 7.67}

    # Per instance weights calculation so we can correctly apply event weights to diagnostic plots
    train_weights = traindataset['EventWeight'].values * traindataset['xsec_rwgt'].values
    test_weights = valdataset['EventWeight'].values * valdataset['xsec_rwgt'].values

    # Fit label encoder to Y_train
    newencoder = LabelEncoder()
    newencoder.fit(Y_train)
    # Transform to encoded array
    encoded_Y = newencoder.transform(Y_train)
    encoded_Y_test = newencoder.transform(Y_test)
    # Transform to one hot encoded arrays
    Y_train = np_utils.to_categorical(encoded_Y)
    Y_test = np_utils.to_categorical(encoded_Y_test)

    optimizer = 'Adam'#'Nadam'
    if do_model_fit == 1:
        histories = []
        labels = []
        # Define model and early stopping
        early_stopping_monitor = EarlyStopping(patience=100,monitor='val_loss',verbose=1)
        model3 = baseline_model(num_variables,optimizer,number_of_classes)

        # Fit the model
        # Batch size = examples before updating weights (larger = faster training)
        # Epochs = One pass over data (useful for periodic logging and evaluation)
        #history3 = model3.fit(X_train,Y_train,validation_split=0.2,epochs=500,batch_size=1000,verbose=1,shuffle=True,class_weight=tuned_weighted,callbacks=[early_stopping_monitor])
        history3 = model3.fit(X_train,Y_train,validation_split=0.2,epochs=300,batch_size=1500,verbose=1,shuffle=True,sample_weight=sampleweights,callbacks=[early_stopping_monitor])
        histories.append(history3)
        labels.append(optimizer)

        # Make plot of loss function evolution
        Plotter.plot_training_progress_acc(histories, labels)
        acc_progress_filename = 'DNN_acc_wrt_epoch.png'
        Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename)
        # Which model do you want the rest of the plots for?
        model = model3
    else:
        # Which model do you want to load?
        model_name = os.path.join(output_directory,'model.h5')
        print '<train-DNN> Loaded Model: %s' % (model_name)
        model = load_trained_model(model_name,num_variables,optimizer,number_of_classes)

    # Node probabilities for training sample events
    result_probs = model.predict(np.array(X_train))
    result_classes = model.predict_classes(np.array(X_train))

    # Node probabilities for testing sample events
    result_probs_test = model.predict(np.array(X_test))
    result_classes_test = model.predict_classes(np.array(X_test))

    # Store model in file
    model_output_name = os.path.join(output_directory,'model.h5')
    model.save(model_output_name)
    weights_output_name = os.path.join(output_directory,'model_weights.h5')
    model.save_weights(weights_output_name)
    model_json = model.to_json()
    model_json_name = os.path.join(output_directory,'model_serialised.json')
    with open(model_json_name,'w') as json_file:
        json_file.write(model_json)
    model.summary()
    model_schematic_name = os.path.join(output_directory,'model_schematic.png')
    plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True)

    # Initialise output directory.
    Plotter.plots_directory = plots_dir
    Plotter.output_directory = output_directory

    # Make overfitting plots of output nodes
    Plotter.overfitting(model, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights)

    # Get true process values for testing dataset
    original_encoded_test_Y = []
    for i in xrange(len(result_probs_test)):
        if Y_test[i][0] == 1:
            original_encoded_test_Y.append(0)
        if Y_test[i][1] == 1:
            original_encoded_test_Y.append(1)
        if Y_test[i][2] == 1:
            original_encoded_test_Y.append(2)
        if Y_test[i][3] == 1:
            original_encoded_test_Y.append(3)

    # Get true process integers for training dataset
    original_encoded_train_Y = []
    for i in xrange(len(result_probs)):
        if Y_train[i][0] == 1:
            original_encoded_train_Y.append(0)
        if Y_train[i][1] == 1:
            original_encoded_train_Y.append(1)
        if Y_train[i][2] == 1:
            original_encoded_train_Y.append(2)
        if Y_train[i][3] == 1:
            original_encoded_train_Y.append(3)

    # Get true class values for testing dataset
    result_classes_test = newencoder.inverse_transform(result_classes_test)
    result_classes_train = newencoder.inverse_transform(result_classes)

    # Create confusion matrices for training and testing performance
    Plotter.conf_matrix(original_encoded_train_Y,result_classes_train,train_weights,'index')
    Plotter.save_plots(dir=plots_dir, filename='yields_norm_confusion_matrix_TRAIN.png')
    Plotter.conf_matrix(original_encoded_test_Y,result_classes_test,test_weights,'index')
    Plotter.save_plots(dir=plots_dir, filename='yields_norm_confusion_matrix_TEST.png')

    Plotter.conf_matrix(original_encoded_train_Y,result_classes_train,train_weights,'')
    Plotter.save_plots(dir=plots_dir, filename='yields_matrix_TRAIN.png')
    Plotter.conf_matrix(original_encoded_test_Y,result_classes_test,test_weights,'')
    Plotter.save_plots(dir=plots_dir, filename='yields_matrix_TEST.png')

    Plotter.ROC_sklearn(original_encoded_train_Y, result_probs, original_encoded_test_Y, result_probs_test, 0 , 'ttHnode')
    Plotter.ROC_sklearn(original_encoded_train_Y, result_probs, original_encoded_test_Y, result_probs_test, 1 , 'Other')
    Plotter.ROC_sklearn(original_encoded_train_Y, result_probs, original_encoded_test_Y, result_probs_test, 2 , 'ttWnode')
    Plotter.ROC_sklearn(original_encoded_train_Y, result_probs, original_encoded_test_Y, result_probs_test, 3 , 'tHQnode')

    # Make table of separation on each node.
    #Plotter.separation_table(Plotter.output_directory)

main()
