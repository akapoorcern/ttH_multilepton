#!/usr/bin/env python

# Select Theano as backend for Keras
import ROOT
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

from ROOT import TMVA, TFile, TString, TLegend, THStack, TTree, TBranch
from array import array
from subprocess import call
from os.path import isfile
# Hack to circumvent error "AttributeError: 'module' object has no attribute 'control_flow_ops'" in current versions of TF and Keras.
# Issue to do with tensorflow removing undocumented symbols from tensorflow/python/__init__.py
# meaning control_flow_ops was no longer visible as submodule of tensorflow.python
# https://github.com/keras-team/keras/issues/3857
import tensorflow as tf
tf.python.control_flow_ops = tf

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
reader = TMVA.Reader("Color:!Silent")

# Check files exist
if not isfile('samples/DiLepTR_ttH_bInclude.root'):
    print 'No such input file: samples/DiLepTR_ttH_bInclude.root'

# Open files and load ttrees
signal_file = TFile.Open('samples/DiLepTR_ttH_bInclude.root')
signal = signal_file.Get('BOOM')

# Define variables list, same as training (order IS important)
variable_list = [('maxeta'),('mindrlep1jet'),('mindrlep2jet'),('SR_InvarMassT'),('corrptlep1'),('corrptlep2'),('hadTop_BDT'),('Hj1_BDT')]

branches = {}

# Register names of inputs with reader. Together with the name give the address of the local variable that carries the updated input variables during event loop.
for branchName in variable_list:
    branches[branchName] = array('f', [1])
    reader.AddVariable(branchName, branches[branchName])
    signal.SetBranchAddress(branchName, branches[branchName])

# Keep track of event numbers for later cross-checks.
event_number = array('f',[0])
signal.SetBranchAddress('EVENT_event', event_number)
reader.AddSpectator('EVENT_event', event_number)

# Define outputs: files to store histograms/ttree with results from application of classifiers and any histos/trees themselves.
output_file_name = 'testsignal_MCDNN_application.root'
output_file = TFile.Open(output_file_name,'RECREATE')
histo_signalnode_signalsample = ROOT.TH1D('histo_signalnode_signalsample','DNN Response ttH sample on ttH node',100,0,1)

# Book methods
# First argument is user defined name. Doesn not have to be same as training name.
# True type of method and full configuration are read from the weights file specified in the second argument.
reader.BookMVA('DNN', TString('MultiClass_DNN/weights/TMVAClassification_DNN.weights.xml'))

# Loop over ttH ttree evaluating MVA as we go.
for i in range(signal.GetEntries()):
    signal.GetEntry(i)
    maxeta = signal.maxeta
    mindrlep1jet = signal.mindrlep1jet
    mindrlep2jet = signal.mindrlep2jet
    SR_InvarMassT = signal.SR_InvarMassT
    corrptlep1 = signal.corrptlep1
    corrptlep2 = signal.corrptlep2
    hadTop_BDT = signal.hadTop_BDT
    Hj1_BDT = signal.Hj1_BDT
    print 'Signal node response: ', reader.EvaluateMulticlass('DNN')[0]
    histo_signalnode_signalsample.Fill(reader.EvaluateMulticlass('DNN')[0])

# Write and close output file.
histo_signalnode_signalsample.Write()
output_file.Close()
