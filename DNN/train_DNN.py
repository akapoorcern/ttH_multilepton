#!/usr/bin/env python
############################################
#              train_DNN.py
#         Joshuha Thomas-Wilsker
#           IHEP Beijing, CERN
############################################
# Usage: python train_DNN.py
############################################
# Python script using Keras with TensorFlow
# backend to train deep neural network for
# ttH multilepton dilepton analysis region.
############################################

# Select TensorFlow as backend for Keras
from os import environ
environ["KERAS_BACKEND"] = "tensorflow"
#environ['KERAS_BACKEND'] = 'theano'
#environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'
import ROOT
from ROOT import TMVA, TFile, TTree, TCut
from array import array
from subprocess import call
from os.path import isfile

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.layers import Dropout

# Hack to circumvent error "AttributeError: 'module' object has no attribute 'control_flow_ops'" in current versions of TF and Keras.
# Issue to do with tensorflow removing undocumented symbols from tensorflow/python/__init__.py
# meaning control_flow_ops was no longer visible as submodule of tensorflow.python
# https://github.com/keras-team/keras/issues/3857
import tensorflow as tf
tf.python.control_flow_ops = tf

def main():

    number_of_hidden_layers = 2

    # Setup TMVA interface to use Keras
    #TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()

    suffix = 'new'
    output_file_name = 'ttHML_MCDNN_%s.root'%str(suffix)
    output_file = TFile.Open(output_file_name,'RECREATE')

    # 'AnalysisType' is where one defines what kind of analysis youre doing e.g. multiclass, Classification ....
    # VarTransform: Decorrelation, PCA-transformation, Gaussianisation, Normalisation (for all classes if none is specified).
    factory = TMVA.Factory('TMVAClassification',output_file,'!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=multiclass')

    #Load data
    input_file_name_signal = 'samples/DiLepTR_ttH_bInclude.root'
    data_signal = TFile.Open(input_file_name_signal)
    signal = data_signal.Get('BOOM')

    input_file_name_ttJets = 'samples/DiLepTR_ttJets_bInclude.root'
    data_bckg_ttJets = TFile.Open(input_file_name_ttJets)
    background_ttJets = data_bckg_ttJets.Get('BOOM')

    input_file_name_ttV = 'samples/DiLepTR_ttV_bInclude.root'
    data_bckg_ttV = TFile.Open(input_file_name_ttV)
    background_ttV = data_bckg_ttV.Get('BOOM')

    # Declare a dataloader interface
    dataloader = TMVA.DataLoader('MultiClass_DNN')

    ### Global event weights ###
    # Can add selection cuts via:
    #dataloader.AddTree(background_ttJets, 'Background_1', 'myvar > cutBarrelOnly && myEventTypeVar=1', backgroundWeight)
    signalWeight = 1.
    backgroundWeight0 = 1.
    backgroundWeight1 = 1.
    dataloader.AddTree(signal, 'ttH', signalWeight)
    dataloader.AddTree(background_ttV, 'ttV', backgroundWeight0)
    dataloader.AddTree(background_ttJets, 'ttJets', backgroundWeight1)

    #variable_list = [('Jet_numLoose','F'), ('maxeta','F'), ('mindrlep1jet','F'), ('mindrlep2jet','F'), ('SR_InvarMassT','F'), ('corrptlep1','F'), ('corrptlep2','F'), ('hadTop_BDT := max(hadTop_BDT,-1.)','F'), ('Hj1_BDT := max(Hj1_BDT,-1.)','F')]
    #variable_list = [('Jet_numLoose','F'), ('maxeta','F'), ('mindrlep1jet','F'), ('mindrlep2jet','F'), ('SR_InvarMassT','F'), ('corrptlep1','F'), ('corrptlep2','F'), ('hadTop_BDT','F'), ('Hj1_BDT','F')]
    variable_list = [('maxeta','F'), ('mindrlep1jet','F'), ('mindrlep2jet','F'), ('SR_InvarMassT','F'), ('corrptlep1','F'), ('corrptlep2','F'), ('hadTop_BDT','F'), ('Hj1_BDT','F')]
    branches = {}
    for var, vartype in variable_list:
        dataloader.AddVariable(var,vartype)
        branches[var] = array('f', [-999])
        branchName = ''
        if 'hadTop_BDT' in var:
            branchName = 'hadTop_BDT'
        elif 'Hj1_BDT' in var:
            branchName = 'Hj1_BDT'
        else:
            branchName = var
        #signal.SetBranchAddress(branchName, branches[branchName])
        #background_ttV.SetBranchAddress(branchName, branches[branchName])
        #background_ttJets.SetBranchAddress(branchName, branches[branchName])

    dataloader.AddSpectator('EVENT_event','F')

    #for event in signal:
        #print 'event.PUWeight: ', event.PUWeight


    # Nominal event weight:
    # event weight = puWgtNom * trigWgtNom * lepSelEffNom * genWgt * xsecWgt (* 0 or 1 depending on if it passes event selection)
    #dataloader.SetWeightExpression("PUWeight*SF_Trigger_2l*SF_Lepton_2l*EVENT_genWeight", "ttH")
    #dataloader.SetWeightExpression("PUWeight*SF_Trigger_2l*SF_Lepton_2l*EVENT_genWeight", "ttV")
    #dataloader.SetWeightExpression("PUWeight*SF_Trigger_2l*SF_Lepton_2l*EVENT_genWeight", "ttJets")
    #dataloader.SetWeightExpression("PUWeight*SF_Trigger_2l*SF_Lepton_2l", "ttH")
    #dataloader.SetWeightExpression("PUWeight*SF_Trigger_2l*SF_Lepton_2l", "ttV")
    #dataloader.SetWeightExpression("PUWeight*SF_Trigger_2l*SF_Lepton_2l", "ttJets")

    # NormMode: Overall renormalisation of event-by-event weights used in training.
    # "NumEvents" = average weight of 1 per event, independantly renormalised for signal and background.
    # "EqualNumEvents" = average weight of 1 per signal event, sum of weights in background equal to sum of weights for signal.
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
    # softmax ensures output values are in range 0-1. Can be used as predicted probabilities.
    activation_function = 'relu'
    model.add(Dense(100, init='glorot_normal', activation=activation_function, input_dim=len(variable_list)))
    model.add(Dense(100, activation=activation_function)) #Always at least 1 hidden layer

    #Randomly set a fraction rate of input units (defined by argument) to 0 at each update during training (helps prevent overfitting).
    #model.add(Dropout(0.2))

    if number_of_hidden_layers > 1:
        model.add(Dense(100, activation=activation_function))
    if number_of_hidden_layers > 2:
        model.add(Dense(100, activation=activation_function))
    if number_of_hidden_layers > 3:
        model.add(Dense(100, activation=activation_function))

    model.add(Dense(3, activation='softmax'))

    # Set loss and optimizer
    # categorical_crossentropy = optimisation algorithm with logarithmic loss function
    # binary_crossentropy
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy',])

    # Store model in file
    model.save('model.h5')
    model.summary()

    # Book methods
    # Choose classifier and define hyperparameters e.g number of epochs, model filename (as chosen above) etc.
    # VarTransform: Decorrelation, PCA-transformation, Gaussianisation, Normalisation (for all classes if none is specified).
    factory.BookMethod(dataloader, TMVA.Types.kPyKeras, "DNN", 'H:!V:VarTransform=D,G:FilenameModel=model.h5:NumEpochs=10:BatchSize=100')
    #factory.BookMethod(dataloader, TMVA.Types.kMLP, "MLP", "!H:!V:NeuronType=tanh:NCycles=1000:HiddenLayers=N+5,5:TestRate=5:EstimatorType=MSE")

    # Run training, testing and evaluation
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

main()
