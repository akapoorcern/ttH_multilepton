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

def main():

    number_of_hidden_layers = 2

    # Setup TMVA interface to use Keras
    #TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()

    output_file_name = 'ttHML_MCDNN_hiddenLayers_%s.root'%str(number_of_hidden_layers)
    output_file = TFile.Open(output_file_name,'RECREATE')

    # 'AnalysisType' is where one defines what kind of analysis youre doing e.g. multiclass, Classification ....
    factory = TMVA.Factory('TMVAClassification',output_file,'!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=multiclass')
    #factory = TMVA.Factory('TMVAClassification', output_file, '!V:!Silent:Color:DrawProgressBar:Transformations=G:AnalysisType=Classification')

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
    #dataloader = TMVA.DataLoader('BDT_ttHML')


    variable_list = [('Jet_numLoose','D'),('maxeta','D'),('mindrlep1jet','D'),('mindrlep2jet','D'),('SR_InvarMassT','D'),('corrptlep1','D'),('corrptlep2','D'),('hadTop_BDT := max(hadTop_BDT,-1.)','D'),('Hj1_BDT := max(Hj1_BDT,-1.)','D')]
    for var, vartype in variable_list:
        dataloader.AddVariable(var,vartype)



    ### Global event weights ###
    # Can add selection cuts via:
    #dataloader.AddTree(background_ttJets, 'Background_1', 'myvar > cutBarrelOnly && myEventTypeVar=1', backgroundWeight)
    signalWeight = 1.
    backgroundWeight0 = 1.
    backgroundWeight1 = 1.
    dataloader.AddTree(signal, 'sample=ttH', signalWeight)
    dataloader.AddTree(background_ttV, 'sample=ttV', backgroundWeight0)
    dataloader.AddTree(background_ttJets, 'sample=ttJets', backgroundWeight1)

    # Set individual event weights (the variables must exist in the original TTree)
    # Nominal event weight:
    # event weight = puWgtNom * trigWgtNom * lepSelEffNom * genWgt * xsecWgt (* 0 or 1 depending on if it passes event selection)
    #dataloader.SetSignalWeightExpression("PUWeight*SF_Trigger_2l*SF_Lepton_2l*EVENT_genWeight*EVENT_xsecWeight")
    #dataloader.SetBackgroundWeightExpression("PUWeight*SF_Trigger_2l*SF_Lepton_2l*EVENT_genWeight*EVENT_xsecWeight")

    # NormMode: Overall renormalisation of event-by-event weights used in training.
    # "NumEvents" = average weight of 1 per event, independantly renormalised for signal and background.
    # "EqualNumEvents" = average weight of 1 per signal event, sum of weights in background equal to sum of weights for signal.
    print 'PrepareTrainingAndTestTree'
    dataloader.PrepareTrainingAndTestTree(TCut(''), 'V:SplitMode=Random:NormMode=EqualNumEvents')

    print 'Generate Model'
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
    print 'Add layers to DNN'
    # softmax ensures output values are in range 0-1. Can be used as predicted probabilities.
    activation_function = 'relu'
    model.add(Dense(300, init='glorot_normal', activation=activation_function, input_dim=len(variable_list)))
    model.add(Dense(300, activation=activation_function)) #Always at least 1 hidden layer

    if number_of_hidden_layers > 1:
        model.add(Dense(300, activation=activation_function))
    if number_of_hidden_layers > 2:
        model.add(Dense(300, activation=activation_function))
    if number_of_hidden_layers > 3:
        model.add(Dense(300, activation=activation_function))

    model.add(Dense(3, activation='softmax'))
    #model.add(Dense(3, activation='sigmoid'))


    # Set loss and optimizer
    # categorical_crossentropy = optimisation algorithm with logarithmic loss function
    print 'compile model'
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy',])

    # Store model in file
    model.save('model.h5')
    model.summary()

    # Book methods
    print 'BookMethod'
    #factory.BookMethod(dataloader,  TMVA.Types.kBDT, "BDTG", "!H:!V:NTrees=1000:BoostType=Grad:Shrinkage=0.10:UseBaggedBoost:BaggedSampleFraction=0.50:nCuts=20:MaxDepth=2")
    factory.BookMethod(dataloader, TMVA.Types.kPyKeras, "DNN", 'H:!V:VarTransform=D,G:FilenameModel=model.h5:NumEpochs=20:BatchSize=32')

    # Run training, testing and evaluation
    print 'Train'
    factory.TrainAllMethods()
    print 'Test'
    factory.TestAllMethods()
    print 'Evaluate'
    factory.EvaluateAllMethods()


main()
