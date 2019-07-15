import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import optparse, json, argparse, math
import ROOT
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import class_weight
from sklearn.metrics import log_loss
import os
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras import backend as K
from keras.utils import np_utils, plot_model
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam, Adadelta, Adagrad
from keras.layers import Input, Dropout, Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from plotting.plotter import plotter
from root_numpy import root2array, tree2array
seed = 7
np.random.seed(7)
rng = np.random.RandomState(31337)

def load_data(inputPath,variables,criteria):
    print variables
    my_cols_list=variables+['process', 'key', 'target', 'totalWeight']
    data = pd.DataFrame(columns=my_cols_list)
    keys=['ttH','ttJ','ttW','ttZ']

    for key in keys :
        print key
        if 'ttH' in key or 'TTH' in key:
            sampleName='ttH'
            fileName = 'ttHnobb_NoJetNCut'
        if 'ttJ' in key or 'TTJ' in key:
            sampleName='ttJ'
            fileName='ttJets_NoJetNCut'
        if 'ttW' in key or 'TTW' in key:
            sampleName='ttW'
            fileName='ttWJets_NoJetNCut'
        if 'ttZ' in key or 'TTZ' in key:
            sampleName='ttZ'
            fileName='ttZJets_NoJetNCut'
        if 'ttH' in key:
                target=0
        if 'ttJ' in key:
                target=1
        if 'ttW' in key:
                target=2
        if 'ttZ' in key:
                target=3

        inputTree = 'syncTree'
        try: tfile = ROOT.TFile(inputPath+"/"+fileName+".root")
        except :
            print " file "+ inputPath+"/"+fileName+".root deosn't exits "
            continue
        try: tree = tfile.Get(inputTree)
        except :
            print inputTree + " deosn't exists in " + inputPath+"/"+fileName+".root"
            continue
        if tree is not None :
            try: chunk_arr = tree2array(tree=tree, selection=criteria) #,  start=start, stop = stop)
            except : continue
            else :
                chunk_df = pd.DataFrame(chunk_arr, columns=variables)
                chunk_df['process']=sampleName
                chunk_df['key']=key
                chunk_df['target']=target
                chunk_df['totalWeight']=chunk_df["EventWeight"]
                data=data.append(chunk_df, ignore_index=True)
        tfile.Close()
        if len(data) == 0 : continue
        nttH = len(data.ix[(data.target.values == 0) & (data.key.values==key) ])
        nttJ = len(data.ix[(data.target.values == 1) & (data.key.values==key) ])
        nttW = len(data.ix[(data.target.values == 2) & (data.key.values==key) ])
        nttZ = len(data.ix[(data.target.values == 3) & (data.key.values==key) ])
        print 'key = ', key
        print "length of nttH = %i, nttJ = %i, nttW = %i, nttZ = %i, TotalWeights = %f" % (nttH, nttJ , nttW, nttZ, data.ix[(data.key.values==key)]["totalWeight"].sum())
        nNW = len(data.ix[(data["totalWeight"].values < 0) & (data.key.values==key) ])
    print (data.columns.values.tolist())
    n = len(data)
    nttH = len(data.ix[data.target.values == 0])
    nttJ = len(data.ix[data.target.values == 1])
    nttW = len(data.ix[data.target.values == 2])
    nttZ = len(data.ix[data.target.values == 3])
    print "Total length of nttH = %i, nttJ = %i, nttW = %i, nttZ = %i" % (nttH, nttJ , nttW, nttZ)
    return data

def load_trained_model(weights_path, num_variables, optimizer):
    model = baseline_model(num_variables, optimizer)
    model.load_weights(weights_path)
    return model

def load_trained_CNN_model(weights_path, lowlevel_num_variables, optimizer, highlevel_num_variables):
    model = functional_CNN_model(lowlevel_num_variables, optimizer, highlevel_num_variables)
    model.load_weights(weights_path)
    return model

def normalise(x_train, x_test):
    mu = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_normalised = (x_train - mu) / std
    x_test_normalised = (x_test - mu) / std
    return x_train_normalised, x_test_normalised

def create_model(num_variables=26, optimizer='Adam', init='glorot_normal'):
    model = Sequential()
    # input_dim = number of variables
    num_variables = 18
    model.add(Dense(32,input_dim=num_variables,kernel_initializer=init,activation='relu'))
    for index in xrange(5):
        model.add(Dense(16,activation='relu'))
    for index in xrange(5):
        model.add(Dense(16,activation='relu'))
    for index in xrange(5):
        model.add(Dense(8,activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def functional_CNN_model(num_variables,optimizer,num_high_level_features):
    # Each example consists of 'num_variables' features.
    # Number of features = spatial dimension. Convolution occurs over this dimension.
    # Each feature is represented by it's 4-vector.
    visible = Input(shape=(num_variables,4))
    # kernel_size can be optimised but can't be larger than the convolution dimension.
    conv_1 = Conv1D(filters=64, kernel_size=2, activation='relu',kernel_initializer='glorot_normal')(visible)
    conv_2 = Conv1D(filters=64, kernel_size=4, activation='relu')(conv_1)
    do_1 = Dropout(0.5)(conv_2)
    # Max pooling layer - used to reduce the complexity of the output and prevent overfitting.
    # Generalises the results from a convolutional feature - detection of features invariant to scale or orientation changes.
    mp_1 = MaxPooling1D(pool_size=2)(do_1)
    #conv_3 = Conv1D(filters=16, kernel_size=5, activation='relu')(mp_1)
    #do_2 = Dropout(0.5)(conv_3)
    #mp_2 = MaxPooling1D(pool_size=2)(do_2)
    conv_conn = Flatten()(mp_1)
    # Merge low-level feature extractor with high-level features.
    #HLFs_ = Input(shape=(num_high_level_features,))
    #merge = concatenate([HLFs_,conv_conn])
    #merge_fully_conn = Dense(32,activation='relu')(merge)
    merge_fully_conn = Dense(32,activation='relu')(conv_conn)
    fully_conn_1 = Dense(16,activation='relu')(merge_fully_conn)
    fully_conn_2 = Dense(16,activation='relu')(fully_conn_1)
    fully_conn_3 = Dense(16,activation='relu')(fully_conn_2)
    fully_conn_4 = Dense(16,activation='relu')(fully_conn_3)
    fully_conn_5 = Dense(16,activation='relu')(fully_conn_4)
    fully_conn_6 = Dense(16,activation='relu')(fully_conn_5)
    fully_conn_7 = Dense(16,activation='relu')(fully_conn_6)
    fully_conn_8 = Dense(16,activation='relu')(fully_conn_7)
    fully_conn_9 = Dense(16,activation='relu')(fully_conn_8)
    fully_conn_10 = Dense(16,activation='relu')(fully_conn_9)
    fully_conn_11 = Dense(8,activation='relu')(fully_conn_10)
    fully_conn_12 = Dense(8,activation='relu')(fully_conn_11)
    fully_conn_13 = Dense(8,activation='relu')(fully_conn_12)
    fully_conn_14 = Dense(8,activation='relu')(fully_conn_13)
    fully_conn_15 = Dense(8,activation='relu')(fully_conn_14)

    # Output layer: 4 nodes.
    output_layer = Dense(4,activation='softmax')(fully_conn_15)
    #model = Model(inputs=[visible,HLFs_], outputs=output_layer)
    model = Model(inputs=visible, outputs=output_layer)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def baseline_model(num_variables,optimizer):
    model = Sequential()
    train_x_shape_dim2 = num_variables
    print 'Conv1D input_shape = (%s, %s)' % (train_x_shape_dim2,1)
    # filters = dimensionality of output space.
    #           Defines the # different filters in this convolution layer.
    #           All filters are of same length, defined by kernel size.
    #
    # kernel_size = specifies length of 1D convolution window that will slide (convolve) over input features.
    #
    # input_shape = number of features per example, feature depth/embedding.
    #               In this case, each feature is defined by a single number so has depth 1.

    # Outputs will have shape (# examples, # features, # filters)
    # Filters will have shape (kernel_size, feature_depth, # filters)
    model.add( Conv1D( filters=64, kernel_size=4, activation='relu', input_shape=(train_x_shape_dim2,1) ) )
    model.add( Conv1D(64, 4, activation='relu') )
    # Dropout layer to reduce chance of over-training.
    model.add(Dropout(0.5))
    # Pooling: reduce dimensionality of problem.
    # Max pooling calculates the maximum value for each patch in the fature map output by the Conv1D layer.
    # Pooling layers will make the network invariant wrt local translations.
    model.add(MaxPooling1D(pool_size=2))
    # Additional pooling layer to avoid overfitting
    #model.add( GlobalAveragePooling1D() )
    # Now flatten. Connection between the convolution and dense layers.
    # Converts N-dimensional arrays into a single continuos linear feature vector.
    model.add(Flatten())
    # Fully connected layers. Don't have the local limitations of convolution layers (they see the whole 'image').
    # Combine local features of previous convolutional layers.
    #for index in xrange(5):
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    # Output layer: 4 nodes.
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def check_dir(dir):
    if not os.path.exists(dir):
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

def reshape_for_particle_rep(dataset, columns):
    # For the low-level variables we want to re-organise into embedded particle representation.
    dataset = dataset[columns]
    #print 'Original dataframe = '
    #print dataset

    jet1_pt_ = dataset['jet1_pt']
    jet1_eta_ = dataset['jet1_eta']
    jet1_phi_ = dataset['jet1_phi']
    jet1_E_ = dataset['jet1_E']
    jet1_ = np.array(zip(jet1_pt_,jet1_eta_,jet1_phi_,jet1_E_))

    jet2_pt_ = dataset['jet2_pt']
    jet2_eta_ = dataset['jet2_eta']
    jet2_phi_ = dataset['jet2_phi']
    jet2_E_ = dataset['jet2_E']
    jet2_ = np.array(zip(jet2_pt_,jet2_eta_,jet2_phi_,jet2_E_))

    jet3_pt_ = dataset['jet3_pt']
    jet3_eta_ = dataset['jet3_eta']
    jet3_phi_ = dataset['jet3_phi']
    jet3_E_ = dataset['jet3_E']
    jet3_ = np.array(zip(jet3_pt_,jet3_eta_,jet3_phi_,jet3_E_))

    jet4_pt_ = dataset['jet4_pt']
    jet4_eta_ = dataset['jet4_eta']
    jet4_phi_ = dataset['jet4_phi']
    jet4_E_ = dataset['jet4_E']
    jet4_ = np.array(zip(jet4_pt_,jet4_eta_,jet4_phi_,jet4_E_))

    lep1_conePt_ = dataset['lep1_conePt']
    lep1_eta_ = dataset['lep1_eta']
    lep1_phi_ = dataset['lep1_phi']
    lep1_E_ = dataset['lep1_E']
    lep1_ = np.array(zip(lep1_conePt_,lep1_eta_,lep1_phi_,lep1_E_))

    lep2_conePt_ = dataset['lep2_conePt']
    lep2_eta_ = dataset['lep2_eta']
    lep2_phi_ = dataset['lep2_phi']
    lep2_E_ = dataset['lep2_E']
    lep2_ = np.array(zip(lep2_conePt_,lep2_eta_,lep2_phi_,lep2_E_))

    reshaped_3D_data = np.array(zip(jet1_,jet2_,jet3_,jet4_,lep1_,lep2_))
    #print 'Reshaped np array'
    #print reshaped_3D_data

    return reshaped_3D_data

def main():
    print 'Using Keras version: ', keras.__version__

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-t', '--train_model', dest='train_model', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=0, type=int)
    parser.add_argument('-r', '--region', dest='region', help='Option to choose SigRegion or CtrlRegion', default='SigRegion', type=str)
    parser.add_argument('-w', '--classweights', dest='classweights', help='Option to choose class weights (InverseNEventsTR, InverseSRYields or BalancedWeights)', default='InverseNEventsTR', type=str)
    parser.add_argument('-s', '--sel', dest='selection', help='Option to choose selection', default='geq4j', type=str)
    args = parser.parse_args()
    do_model_fit = args.train_model
    region = args.region
    classweights_name = args.classweights
    selection = args.selection
    number_of_classes = 4

    # Create instance of output directory where all results are saved.
    output_directory = '2019-06-18_CNN_LLFOnly_FunkAPI_particleinput_%s_%s_%s/' % (selection,classweights_name,region)

    check_dir(output_directory)

    # Create plots subdirectory
    plots_dir = os.path.join(output_directory,'plots/')

    lowlevel_invar_jsonFile = open('LOWLEVEL_invars_conv1DNN.json','r')
    highlevel_invar_jsonFile = open('HIGHLEVEL_invars_conv1DNN.json','r')

    if selection == 'geq4j':
        selection_criteria = 'Jet_numLoose>=4'
    if selection == 'geq3j':
        selection_criteria = 'Jet_numLoose>=3'
    if selection == 'eeq3j':
        selection_criteria = 'Jet_numLoose==3'

    # WARNINING !!!!
    #variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()
    lowlevel_invar_list = json.load(lowlevel_invar_jsonFile,encoding="utf-8").items()
    highlevel_invar_list = json.load(highlevel_invar_jsonFile,encoding="utf-8").items()

    lowlevel_column_headers = []
    for key,var in lowlevel_invar_list:
        lowlevel_column_headers.append(key)
    lowlevel_column_headers.append('EventWeight')
    lowlevel_column_headers.append('xsec_rwgt')

    highlevel_column_headers = []
    for key,var in highlevel_invar_list:
        if 'hadTop_BDT' in key:
            key = 'hadTop_BDT'
        if 'Hj1_BDT' in key:
            key = 'Hj1_BDT'
        if 'Hj_tagger_hadTop' in key:
            key = 'Hj_tagger_hadTop'
        highlevel_column_headers.append(key)
    highlevel_column_headers.append('EventWeight')
    highlevel_column_headers.append('xsec_rwgt')

    # Create instance of the input files directory
    inputs_file_path = '/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/keras-DNN/samples/Training_samples_looselepsel/'

    print 'Getting files from:', inputs_file_path
    lowlevel_features_DF_name = '%s/lowlevel_features_DF_%s_%s.csv' %(output_directory,region,selection)
    highlevel_features_DF_name = '%s/highlevel_features_DF_%s_%s.csv' %(output_directory,region,selection)

    if os.path.isfile(lowlevel_features_DF_name):
        lowlevel_features_data = pandas.read_csv(lowlevel_features_DF_name)
        print 'Loading %s . . . . ' % (lowlevel_features_DF_name)
    else:
        print 'Creating and loading new data file in %s . . . . ' % (inputs_file_path)
        lowlevel_features_data = load_data(inputs_file_path,lowlevel_column_headers,selection_criteria)
        lowlevel_features_data.to_csv(lowlevel_features_DF_name, index=False)
        lowlevel_features_data = pandas.read_csv(lowlevel_features_DF_name)

    if os.path.isfile(highlevel_features_DF_name):
        highlevel_features_data = pandas.read_csv(highlevel_features_DF_name)
        print 'Loading %s . . . . ' % (highlevel_features_DF_name)
    else:
        print 'Creating and loading new data file in %s . . . . ' % (inputs_file_path)
        highlevel_features_data = load_data(inputs_file_path,highlevel_column_headers,selection_criteria)
        highlevel_features_data.to_csv(highlevel_features_DF_name, index=False)
        highlevel_features_data = pandas.read_csv(highlevel_features_DF_name)

    Plotter = plotter()

    # Split pandas dataframe into train/test
    lowlevel_traindataset, lowlevel_valdataset = train_test_split(lowlevel_features_data, test_size=0.2)
    highlevel_traindataset, highlevel_valdataset = train_test_split(highlevel_features_data, test_size=0.2)

    print 'LOWLEVEL train dataset shape [ Nexamples: %s , Nfeatures: %s ]' % (lowlevel_traindataset.shape[0], lowlevel_traindataset.shape[1])
    print 'LOWLEVEL validation dataset shape [ Nexamples: %s , Nfeatures: %s ]' % (lowlevel_valdataset.shape[0], lowlevel_valdataset.shape[1])
    print 'HIGHLEVEL train dataset shape [ Nexamples: %s , Nfeatures: %s ]' % (highlevel_traindataset.shape[0], highlevel_traindataset.shape[1])
    print 'HIGHLEVEL validation dataset shape [ Nexamples: %s , Nfeatures: %s ]' % (highlevel_valdataset.shape[0], highlevel_valdataset.shape[1])

    #feature_corr_df = lowlevel_traindataset + highlevel_traindataset
    lowlevel_train_df = lowlevel_features_data.iloc[:lowlevel_traindataset.shape[0]]
    lowlevel_train_df.drop(['EventWeight'], axis=1, inplace=True)
    lowlevel_train_df.drop(['xsec_rwgt'], axis=1, inplace=True)
    highlevel_train_df = highlevel_features_data.iloc[:highlevel_traindataset.shape[0]]
    highlevel_train_df.drop(['EventWeight'], axis=1, inplace=True)
    highlevel_train_df.drop(['xsec_rwgt'], axis=1, inplace=True)

    # calculate event weights
    train_weights = lowlevel_traindataset['EventWeight'].values * lowlevel_traindataset['xsec_rwgt'].values
    test_weights = lowlevel_valdataset['EventWeight'].values * lowlevel_valdataset['xsec_rwgt'].values

    # Remove unwanted variables from columns list use for training
    lowlevel_training_columns = lowlevel_column_headers[:-2]
    highlevel_training_columns = highlevel_column_headers[:-2]

    # Collect just the values for the variables used in the training and testing data sets.
    lowlevel_X_train = lowlevel_traindataset[lowlevel_training_columns].values
    lowlevel_X_test = lowlevel_valdataset[lowlevel_training_columns].values

    reshaped_3D_data = reshape_for_particle_rep(lowlevel_traindataset , lowlevel_training_columns)

    reshaped_3D_data_test = reshape_for_particle_rep(lowlevel_valdataset , lowlevel_training_columns)

    highlevel_X_train = highlevel_traindataset[highlevel_training_columns].values
    highlevel_X_test = highlevel_valdataset[highlevel_training_columns].values

    Y_train = lowlevel_traindataset.target.astype(int)
    Y_test = lowlevel_valdataset.target.astype(int)

    # Need to reshape data to have spatial dimension for conv1d
    lowlevel_X_train = np.expand_dims(lowlevel_X_train, axis=-1)
    lowlevel_X_test = np.expand_dims(lowlevel_X_test, axis=-1)
    #print 'Reshaped lowlevel_data to include spatial dimension for conv1d. New shape = ', lowlevel_X_train.shape

    lowlevel_num_variables = len(lowlevel_training_columns)
    highlevel_num_variables = len(highlevel_training_columns)

    ## Input Variable Correlations
    lowlevel_correlation_plot_file_name = 'lowlevel_correlation_plot.png'
    #Plotter.correlation_matrix(lowlevel_train_df)
    #Plotter.save_plots(dir=plots_dir, filename=lowlevel_correlation_plot_file_name)

    highlevel_correlation_plot_file_name = 'highlevel_correlation_plot.png'
    #Plotter.correlation_matrix(highlevel_train_df)
    #Plotter.save_plots(dir=plots_dir, filename=highlevel_correlation_plot_file_name)

    # =============== Weights ==================
    # WARNING! 'sample_weight' will overide 'class_weight'
    # ==========================================
    # Sample                    |       ttH       |      tt+jets       |       ttW        |       ttZ        |
    ############################
    #  ======= geq 4 Jets ======
    ############################
    # Loose lepton TR selection
    ############################
    # XS                              0.2118              831.                0.2043            0.2529
    # # events in TR            |     221554      |      1168897       |      321674      |      204998      |
    # Sum of weights:           |    94.379784    |   7372.112793      |    206.978439    |    122.834419    |
    # Yields 2LSS SR HIG 18-019 |      60.08      | 140.25+22.79+17.25 |      151.03      |      87.05       |
    #                                                    =180.29
    ############################
    #= Control Region (== 3 Jets)
    ############################
    # Loose lepton TR selection
    ############################
    # # events in TR            |    39418        |      568724        |      111809      |      58507       |
    # Sum of weights            |   24.269867     |    3807.655762     |    102.885391    |     58.825554    |
    # Yields 2LSS ttWctrl       |    14.36        |    120.54 + 9.55   |       75.97      |      38.64       |
    #   AN2018-098-v18

    # Yields 2LSS SR HIG 18-019 |       60.08        | 140.25+22.79+17.25 |       151.03       |      87.05       |
    # Yields 2LSS ttWctrl       |       14.36        |    120.54 + 9.55   |        75.97       |      38.64       |
    # Yields 2LSS >= 3 jets     |       74.44        |        310.38      |       227.00       |     125.69       |

    if classweights_name == 'InverseSRYields':
        if selection == 'geq4j':
            tuned_weighted = {0 : 0.0166445, 1 : 0.00554662, 2 : 0.00662120, 3 : 0.0114877}
        if selection == 'geq3j':
            tuned_weighted = {0 : 0.01343363782, 1 : 0.00322185707, 2 : 0.00440528634, 3 : 0.00795608242}
    elif classweights_name == 'InverseNEventsTR':
        tuned_weighted = {0 : 0.00000451357, 1 : 0.000000855507, 2 : 0.00000310874, 3 : 0.00000487810}
    elif classweights_name == 'InverseSumWeightsTR':
        tuned_weighted = {0 : 0.01059548939, 1 : 0.00013564632, 2 : 0.00483142111, 3 : 0.00814104066}

    print 'class weights : ', classweights_name

    # Fit label encoder to Y_train
    newencoder = LabelEncoder()
    newencoder.fit(Y_train)
    # Transform to encoded array
    encoded_Y = newencoder.transform(Y_train)
    encoded_Y_test = newencoder.transform(Y_test)
    # Transform to one hot encoded arrays
    Y_train = np_utils.to_categorical(encoded_Y)
    Y_test = np_utils.to_categorical(encoded_Y_test)

    #print 'num_variables = ',num_variables
    optimizer = 'Adam'
    if do_model_fit == 1:

        # Training new model
        histories = []
        labels = []
        early_stopping_monitor = EarlyStopping(patience=50,monitor='val_loss',verbose=1)

        # Lists for HP scan
        #optimizers = ['Adamax','Adam','Nadam']
        #batchessize = np.array([100,200,500,1000])

        # Define a model
        #model3 = baseline_model(lowlevel_num_variables, optimizer)
        print 'Low-level data shape:'
        print reshaped_3D_data.shape
        model4 = functional_CNN_model(reshaped_3D_data.shape[1], optimizer, highlevel_num_variables)

        # Fit the model using training data.
        # Batch size = number of examples before updating weights (larger = faster training)
        #history4 = model4.fit([reshaped_3D_data,highlevel_X_train],Y_train,validation_split=0.2,epochs=200,batch_size=1000,verbose=1,shuffle=True,class_weight=tuned_weighted,callbacks=[early_stopping_monitor])
        history4 = model4.fit(reshaped_3D_data,Y_train,validation_split=0.2,epochs=200,batch_size=1000,verbose=1,shuffle=True,class_weight=tuned_weighted,callbacks=[early_stopping_monitor])

        # Store history for performance by epoch plot.
        histories.append(history4)
        labels.append(optimizer)
        Plotter.plot_training_progress_acc(histories, labels)
        acc_progress_filename = 'DNN_acc_wrt_epoch.png'
        Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename)

        # Which model do you want the rest of the plots for?
        #model = model3
        model = model4
    else:
        # Which model do you want to load?
        model_name = os.path.join(output_directory,'model.h5')
        print 'Loading  %s' % (model_name)
        #model = load_trained_model(model_name, num_variables, optimizer)
        model = load_trained_CNN_model(model_name, reshaped_3D_data.shape[1], optimizer, highlevel_num_variables)

    # Node probabilities for training sample events
    # Is this the same as in the DNN case?
    result_probs_train = model.predict([reshaped_3D_data,highlevel_X_train])
    # Get maximum probability
    result_classes_train = result_probs_train.argmax(axis=-1)

    # Node probabilities for testing sample events
    result_probs_test = model.predict([reshaped_3D_data_test,highlevel_X_train])
    result_classes_test = result_probs_test.argmax(axis=-1)

    # Store model in hdf5 format
    model_output_name = os.path.join(output_directory,'model.h5')
    model.save(model_output_name)

    # Save model weights only seperately as well in hdf5 format
    weights_output_name = os.path.join(output_directory,'model_weights.h5')
    model.save_weights(weights_output_name)

    # Make sure to save model in json format as well
    model_json = model.to_json()
    model_json_name = os.path.join(output_directory,'model_serialised.json')
    with open(model_json_name,'w') as json_file:
        json_file.write(model_json)

    model.summary()

    model_schematic_name = os.path.join(output_directory,'model_schematic.png')
    #plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True)

    # Initialise output directory where plotter results will be saved.
    Plotter.output_directory = output_directory

    # Make overfitting plots
    #Plotter.overfitting(model, Y_train, Y_test, result_probs_train, result_probs_test, plots_dir, train_weights, test_weights)

    # Make list of true labels e.g. (0,1,2,3)
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

    original_encoded_train_Y = []
    for i in xrange(len(result_probs_train)):
        if Y_train[i][0] == 1:
            original_encoded_train_Y.append(0)
        if Y_train[i][1] == 1:
            original_encoded_train_Y.append(1)
        if Y_train[i][2] == 1:
            original_encoded_train_Y.append(2)
        if Y_train[i][3] == 1:
            original_encoded_train_Y.append(3)

    # Invert LabelEncoder transform back to original truth labels
    result_classes_train = newencoder.inverse_transform(result_classes_train)
    result_classes_test = newencoder.inverse_transform(result_classes_test)

    Plotter.plots_directory = plots_dir

    # Create confusion matrices for training and testing performance
    #Plotter.conf_matrix(original_encoded_train_Y,result_classes_train,train_weights,'index')
    #Plotter.save_plots(dir=plots_dir, filename='yields_norm_confusion_matrix_TRAIN.png')
    #Plotter.conf_matrix(original_encoded_test_Y,result_classes_test,test_weights,'index')
    #Plotter.save_plots(dir=plots_dir, filename='yields_norm_confusion_matrix_TEST.png')

    Plotter.ROC_sklearn(original_encoded_train_Y, result_probs_train, original_encoded_test_Y, result_probs_test, 0 , 'ttHnode')
    Plotter.ROC_sklearn(original_encoded_train_Y, result_probs_train, original_encoded_test_Y, result_probs_test, 1 , 'ttJnode')
    Plotter.ROC_sklearn(original_encoded_train_Y, result_probs_train, original_encoded_test_Y, result_probs_test, 2 , 'ttWnode')
    Plotter.ROC_sklearn(original_encoded_train_Y, result_probs_train, original_encoded_test_Y, result_probs_test, 3 , 'ttZnode')

    #Plotter.separation_table(Plotter.output_directory)

main()
