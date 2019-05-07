#!/usr/bin/env python
############################################
#              train_DNN.py
#         Joshuha Thomas-Wilsker
#           IHEP Beijing, CERN
############################################
# python train_DNN.py -s <relative_path_to_signal_sample/sample>.root -x <relative_path_to_bckg1_sample/sample>.root -y <relative_path_to_bckg2_sample/sample>.root -a <activation_function> -l <number_of_hidden_layers> -j <variables_list>.json
############################################
# Python script using Keras with TensorFlow
# backend to train deep neural network for
# ttH multilepton dilepton analysis region.
############################################

# Select TensorFlow as backend for Keras
import os
import ROOT
import optparse
import json
import h5py

from os import environ
environ["KERAS_BACKEND"] = "tensorflow"
#environ['KERAS_BACKEND'] = 'theano'
#environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'
from ROOT import TMVA, TFile, TTree, TCut, TString
from array import array
from subprocess import call
from os.path import isfile
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.layers import Dropout

def main():

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--signal_sample',        dest='input_file_name_signal'  ,      help='signal sample path',      default='samples/samples-NJet_geq_3/MVATraining/ttHnobb_TrainMVANoCutJetN.root',        type='string')
    parser.add_option('-x', '--bckg1_sample',        dest='input_file_name_ttJets'  ,      help='background sample 1 path',      default='samples/samples-NJet_geq_3/MVATraining/ttJets_TrainMVANoCutJetN.root',        type='string')
    parser.add_option('-y', '--bckg2_sample',        dest='input_file_name_ttW'  ,      help='background sample 2 path',      default='samples/samples-NJet_geq_3/MVATraining/ttWJets_TrainMVANoCutJetN.root',        type='string')
    parser.add_option('-z', '--bckg3_sample',        dest='input_file_name_ttZ'  ,      help='background sample 3 path',      default='samples/samples-NJet_geq_3/MVATraining/ttZJets_TrainMVANoCutJetN.root',        type='string')
    parser.add_option('-a', '--activation',        dest='activation_function'  ,      help='activation function',      default='relu',        type='string')
    parser.add_option('-l', '--hidden_layers',        dest='number_of_hidden_layers'  ,      help='number of hidden layers',      default='2',        type='int')
    parser.add_option('-t', '--var_transform',        dest='var_transform_name'  ,      help='transformation used on input variables',      default='None',        type='string')
    parser.add_option('-j', '--json',        dest='json'  ,      help='json file with list of variables',      default=None,        type='string')
    parser.add_option('-r', '--learning_rate',        dest='learning_rate'  ,      help='learning rate',      default=0.008,        type='float')
    parser.add_option('-n', '--num_epochs',        dest='num_epochs'  ,      help='number of epochs',      default=10,        type='string')

    (opt, args) = parser.parse_args()

    number_of_hidden_layers = opt.number_of_hidden_layers
    activation_function = opt.activation_function
    var_transform_name = opt.var_transform_name
    num_epochs = opt.num_epochs
    jsonFile = open(opt.json,'r')
    new_variable_list = json.load(jsonFile,encoding='utf-8').items()
    learning_rate = opt.learning_rate
    layer_nodes = 40

    # Setup TMVA interface to use Keras
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()

    if ',' in var_transform_name:
        var_transform_name_list = var_transform_name.split(',')
        new_var_transform_name = '+'.join(var_transform_name_list)
        print 'new_var_transform_name: ', new_var_transform_name
    else:
        print 'var_transform_name = ', var_transform_name
        new_var_transform_name = var_transform_name
        print 'new_var_transform_name: ' , new_var_transform_name

    num_inputs = 0
    for key, value in new_variable_list:
        num_inputs = num_inputs + 1
    print 'num inputs = ' , str(num_inputs)
    classifier_parent_dir = 'DNN_noCutJetN_%sVars_%sHLs_%s_%s-VarTrans_%s-learnRate_%s-epochs-%s-nodes' % (str(num_inputs),str(number_of_hidden_layers),activation_function,new_var_transform_name,str(learning_rate),num_epochs,str(layer_nodes))
    classifier_samples_dir = classifier_parent_dir+"/outputs"
    if not os.path.exists(classifier_samples_dir):
        os.makedirs(classifier_samples_dir)

    output_file_name = '%s/%s.root'%(classifier_samples_dir,classifier_parent_dir)
    output_file = TFile.Open(output_file_name,'RECREATE')

    # 'AnalysisType' is where one defines what kind of analysis you're doing e.g. multiclass, Classification ....
    # VarTransform: Decorrelation, PCA-transformation, Gaussianisation, Normalisation (for all classes if none is specified).
    # When transformation is specified in factory object, the transformation is only used for informative purposes (not used for classifier inputs).
    # Distributions can be found in output to see how variables would look if transformed.
    factory_name = 'Factory_%s' % (classifier_parent_dir)
    factory_string = '!V:!Silent:Color:DrawProgressBar:Transformations=%s:AnalysisType=multiclass' % var_transform_name
    factory = TMVA.Factory(factory_name, output_file,factory_string)

    #Load data
    input_file_name_signal = opt.input_file_name_signal
    data_signal = TFile.Open(input_file_name_signal)
    signal = data_signal.Get('syncTree')

    input_file_name_ttJets = opt.input_file_name_ttJets
    data_bckg_ttJets = TFile.Open(input_file_name_ttJets)
    background_ttJets = data_bckg_ttJets.Get('syncTree')

    input_file_name_ttW = opt.input_file_name_ttW
    data_bckg_ttW = TFile.Open(input_file_name_ttW)
    background_ttW = data_bckg_ttW.Get('syncTree')

    input_file_name_ttZ = opt.input_file_name_ttZ
    data_bckg_ttZ = TFile.Open(input_file_name_ttZ)
    background_ttZ = data_bckg_ttZ.Get('syncTree')

    # Declare a dataloader interface
    dataloader_name = classifier_parent_dir
    dataloader = TMVA.DataLoader(dataloader_name)

    # Can add selection cuts via:
    # dataloader.AddTree(background_ttJets, 'Background_1', 'myvar > cutBarrelOnly && myEventTypeVar=1', backgroundWeight)

    ### Global event weights ###
    signalWeight = 1.
    backgroundWeight0 = 1.
    backgroundWeight1 = 1.
    backgroundWeight2 = 1.
    dataloader.AddTree(signal, 'ttH', signalWeight)
    dataloader.AddTree(background_ttW, 'ttW', backgroundWeight0)
    dataloader.AddTree(background_ttZ, 'ttZ', backgroundWeight1)
    dataloader.AddTree(background_ttJets, 'ttJets', backgroundWeight2)

    branches = {}
    for key, value in new_variable_list:
        dataloader.AddVariable(str(key))
        branches[key] = array('f', [-999])
        print 'variable: ', key
        branchName = ''
        branchName = key
    dataloader.AddSpectator('nEvent','F')

    # Nominal event weight:
    # event weight = puWgtNom * trigWgtNom * lepSelEffNom * genWgt * xsecWgt (* 0 or 1 depending on if it passes event selection)

    dataloader.SetWeightExpression("EventWeight", "ttH")
    dataloader.SetWeightExpression("EventWeight", "ttW")
    dataloader.SetWeightExpression("EventWeight", "ttZ")
    dataloader.SetWeightExpression("EventWeight", "ttJets")

    # NormMode: Overall renormalisation of event-by-event weights used in training.
    # "NumEvents" = average weight of 1 per event, independantly renormalised for signal and background.
    # "EqualNumEvents" = average weight of 1 per signal event, sum of weights in background equal to sum of weights for signal.
    #dataloader.PrepareTrainingAndTestTree(TCut(''), 'V:NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:NormMode=EqualNumEvents')
    dataloader.PrepareTrainingAndTestTree(TCut(''), 'V:SplitMode=Random:NormMode=EqualNumEvents')

    # Generate model:
    model = Sequential()

    # Add layers to DNN
    '''
    Dense:
    # Number of nodes
    init= # Initialisation
    activation= # Activation
    input_dim= # Shape of inputs (Number of inputs). Argument only needed for first layer.
    '''

    # first hidden layer
    model.add(Dense(layer_nodes, init='glorot_normal', activation=activation_function, input_dim=len(new_variable_list)))

    # Randomly set a fraction rate of input units (defined by argument) to 0 at each update during training (helps prevent overfitting).
    #model.add(Dropout(0.2))


    # Hidden layers
    for x in xrange(number_of_hidden_layers):
        model.add(Dense(layer_nodes, activation=activation_function))

    # Output layer
    # softmax ensures output values are in range 0-1. Can be used as predicted probabilities.
    # 'softmax' activation function used in final layer so that the outputs represent probabilities (output is normalised to 1).
    model.add(Dense(4, activation='softmax'))

    # Set loss and optimizer
    # categorical_crossentropy = optimisation algorithm with logarithmic loss function
    # binary_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learning_rate), metrics=['accuracy',])

    # Store model in file
    model.save('model.h5')
    model.summary()

    # Book methods
    # Choose classifier and define hyperparameters e.g number of epochs, model filename (as chosen above) etc.
    # VarTransform: Decorrelate, PCA, Gauss, Norm, None.
    # Transformations used in booking are used for actual training.
    logs_dir = classifier_parent_dir+'/logs'
    #factory_string_bookMethod = 'H:!V:VarTransform=%s:FilenameModel=model.h5:NumEpochs=%s:BatchSize=100:Tensorboard=%s' % (var_transform_name, num_epochs, logs_dir)
    factory_string_bookMethod = 'H:!V:VarTransform=%s:FilenameModel=model.h5:NumEpochs=%s:BatchSize=100' % (var_transform_name, num_epochs)
    factory.BookMethod(dataloader, TMVA.Types.kPyKeras, "DNN", factory_string_bookMethod)

    # Run training, testing and evaluation
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

main()
