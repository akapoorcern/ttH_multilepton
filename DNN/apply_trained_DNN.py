#!/usr/bin/env python

# Select Theano as backend for Keras
import ROOT
import os
import sys
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
# Set architecture of system (AVX instruction set is not supported on SWAN)
#environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'
from ROOT import TMVA, TFile, TString, TLegend, THStack, TTree, TBranch
from array import array
from subprocess import call
from os.path import isfile
import optparse
import json
import math

def network_evaluation(sample_ttree, variables_list, sample_name, branches_ttree, branches_reader, tmvareader):

    print 'Evaluating %s sample ' % sample_name

    histo_ttHclassified_events_name = 'histo_ttHclassified_events_%s' % sample_name
    histo_ttVclassified_events_name = 'histo_ttVclassified_events_%s' % sample_name
    histo_ttJclassified_events_name = 'histo_ttJclassified_events_%s' % sample_name

    histo_ttHclassified_events_title = 'ttH Classified Events: %s Sample' % sample_name
    histo_ttVclassified_events_title = 'ttV Classified Events: %s Sample' % sample_name
    histo_ttJclassified_events_title = 'ttJ Classified Events: %s Sample' % sample_name

    histo_ttHclassified_events = ROOT.TH1D(histo_ttHclassified_events_name,histo_ttHclassified_events_title,40,0,1.)
    histo_ttVclassified_events = ROOT.TH1D(histo_ttVclassified_events_name,histo_ttVclassified_events_title,40,0,1.)
    histo_ttJclassified_events = ROOT.TH1D(histo_ttJclassified_events_name,histo_ttJclassified_events_title,40,0,1.)

    temp_percentage_done = 0
    for i in range(sample_ttree.GetEntries()):
        percentage_done = int(100*float(i)/float(sample_ttree.GetEntries()))
        if percentage_done % 10 == 0:
            if percentage_done != temp_percentage_done:
                print percentage_done
                temp_percentage_done = percentage_done
        sample_ttree.GetEntry(i)

        for key, value in variables_list:
            branches_reader[str(key)][0] = branches_ttree[str(key)][0]

        event_num = array('d',[0])
        event_num = sample_ttree.EVENT_event

        event_classification = max(tmvareader.EvaluateMulticlass('DNN')[0],tmvareader.EvaluateMulticlass('DNN')[1],tmvareader.EvaluateMulticlass('DNN')[2])
        if event_classification == tmvareader.EvaluateMulticlass('DNN')[0]:
            histo_ttHclassified_events.Fill(event_classification)
        elif event_classification == tmvareader.EvaluateMulticlass('DNN')[1]:
            histo_ttVclassified_events.Fill(event_classification)
        elif event_classification == tmvareader.EvaluateMulticlass('DNN')[2]:
            histo_ttJclassified_events.Fill(event_classification)

    histo_ttHclassified_events.Write()
    histo_ttVclassified_events.Write()
    histo_ttJclassified_events.Write()

def main():

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify inputs from network training',      default=None,        type='string')
    parser.add_option('-j', '--json',        dest='json'  ,      help='json file with list of variables',      default=None,        type='string')

    (opt, args) = parser.parse_args()
    jsonFile = open(opt.json,'r')

    if opt.json == None:
        print 'input variable .json not defined!'
        sys.exit(1)
    if opt.input_suffix == None:
        print 'Input files suffix not defined!'
        sys.exit(1)

    new_variable_list = json.load(jsonFile,encoding='utf-8').items()

    classifier_suffix = opt.input_suffix
    classifier_parent_dir = 'MultiClass_DNN_%s' % (classifier_suffix)

    # Setup TMVA
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()
    reader = TMVA.Reader("Color:!Silent")

    # Check files exist
    if not isfile('samples/DiLepSR_ttH_bInclude.root'):
        print 'No such input file: samples/DiLepSR_ttH_bInclude.root'
    if not isfile('samples/DiLepSR_ttJets_bInclude.root'):
        print 'No such input file: samples/DiLepSR_ttJets_bInclude.root'
    if not isfile('samples/DiLepSR_ttV_bInclude.root'):
        print 'No such input file: samples/DiLepSR_ttV_bInclude.root'

    # Open files and load ttrees
    data_ttH = TFile.Open('samples/DiLepSR_ttH_bInclude.root')
    data_ttV = TFile.Open('samples/DiLepSR_ttV_bInclude.root')
    data_ttJets = TFile.Open('samples/DiLepSR_ttJets_bInclude.root')
    data_ttH_tree = data_ttH.Get('BOOM')
    data_ttV_tree = data_ttV.Get('BOOM')
    data_ttJets_tree = data_ttJets.Get('BOOM')

    branches_doubles = {}
    for key, value in new_variable_list:
        branches_doubles[key] = array('d', [-999])
        if 'hadTop_BDT' in key:
            keyname = 'hadTop_BDT'
            data_ttH_tree.SetBranchAddress(str(keyname), branches_doubles[key])
            data_ttV_tree.SetBranchAddress(str(keyname), branches_doubles[key])
            data_ttJets_tree.SetBranchAddress(str(keyname), branches_doubles[key])
        elif 'Hj1_BDT' in key:
            keyname = 'Hj1_BDT'
            data_ttH_tree.SetBranchAddress(str(keyname), branches_doubles[key])
            data_ttV_tree.SetBranchAddress(str(keyname), branches_doubles[key])
            data_ttJets_tree.SetBranchAddress(str(keyname), branches_doubles[key])
        else:
            data_ttH_tree.SetBranchAddress(str(key), branches_doubles[key])
            data_ttV_tree.SetBranchAddress(str(key), branches_doubles[key])
            data_ttJets_tree.SetBranchAddress(str(key), branches_doubles[key])

    # Keep track of event numbers for cross-checks.
    event_number_branch = array('d',[-999])
    data_ttH_tree.SetBranchAddress('EVENT_event', event_number_branch)

    branches_floats = {}
    # Register names of inputs with reader. Together with the name give the address of the local variable that carries the updated input variables during event loop.
    for key, value in new_variable_list:
        print 'Add variable name %s: ' % key
        branches_floats[key] = array('f', [-999])
        reader.AddVariable(str(key), branches_floats[key])

    event_number = array('f',[-999])
    reader.AddSpectator('EVENT_event', event_number)

    # Book methods
    # First argument is user defined name. Doesn not have to be same as training name.
    # True type of method and full configuration are read from the weights file specified in the second argument.
    mva_weights_dir = '%s/weights/Factory_MultiClass_DNN_%s_DNN.weights.xml' % (classifier_parent_dir,classifier_suffix)
    print 'using weights file: ', mva_weights_dir
    reader.BookMVA('DNN', TString(mva_weights_dir))

    classifier_samples_dir = classifier_parent_dir+"/outputs"
    classifier_plots_dir = classifier_parent_dir+"/plots"
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)
    if not os.path.exists(classifier_samples_dir):
        os.makedirs(classifier_samples_dir)

    # Define outputs: files to store histograms/ttree with results from application of classifiers and any histos/trees themselves.
    output_file_name = '%s/Applied_%s.root' % (classifier_samples_dir,classifier_parent_dir)
    output_file = TFile.Open(output_file_name,'RECREATE')

    # Loop over ttH ttree evaluating MVA as we go.
    # Keep track of the response and input values assigned to every event number for later checks.
    network_evaluation(data_ttH_tree, new_variable_list, 'ttH', branches_doubles, branches_floats, reader)
    network_evaluation(data_ttV_tree, new_variable_list, 'ttV', branches_doubles, branches_floats, reader)
    network_evaluation(data_ttJets_tree, new_variable_list, 'ttJets', branches_doubles, branches_floats, reader)

    output_file.Close()

main()
