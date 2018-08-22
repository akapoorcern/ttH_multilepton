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

def network_evaluation(sample_ttree, variables_list, sample_name, branches_ttree, integer_branches_tree, branches_reader, tmvareader, categorise):

    print 'Evaluating %s sample ' % sample_name

    if categorise == True:
        histoname_type = 'Category'
    else:
        histoname_type = 'Node'

    histo_ttHclassified_events_name = 'histo_ttH%s_events_%s' % (histoname_type,sample_name)
    histo_ttVclassified_events_name = 'histo_ttV%s_events_%s' % (histoname_type,sample_name)
    histo_ttJclassified_events_name = 'histo_ttJ%s_events_%s' % (histoname_type,sample_name)

    histo_ttHclassified_events_title = 'ttH %s Events: %s Sample' % (histoname_type,sample_name)
    histo_ttVclassified_events_title = 'ttV %s Events: %s Sample' % (histoname_type,sample_name)
    histo_ttJclassified_events_title = 'ttJ %s Events: %s Sample' % (histoname_type,sample_name)

    histo_ttHclassified_events = ROOT.TH1D(histo_ttHclassified_events_name,histo_ttHclassified_events_title,20,0,1.)
    histo_ttVclassified_events = ROOT.TH1D(histo_ttVclassified_events_name,histo_ttVclassified_events_title,20,0,1.)
    histo_ttJclassified_events = ROOT.TH1D(histo_ttJclassified_events_name,histo_ttJclassified_events_title,20,0,1.)

    temp_percentage_done = 0
    for i in range(sample_ttree.GetEntries()):
    #for i in range(100):
        percentage_done = int(100*float(i)/float(sample_ttree.GetEntries()))
        if percentage_done % 10 == 0:
            if percentage_done != temp_percentage_done:
                print percentage_done
                temp_percentage_done = percentage_done
        sample_ttree.GetEntry(i)

        for key, value in variables_list:
            if ('hadTop_BDT' in key) or ('Hj1_BDT' in key):
                #print '%s: %f' % (key, max(branches_ttree[str(key)][0],-1) )
                branches_reader[str(key)][0] = max(branches_ttree[str(key)][0],-1)
            elif ('n_fakeablesel_mu' in key) or ('n_fakeablesel_ele' in key) or ('Jet_numLoose' in key):
                #print '%s: %i' % (key, integer_branches_tree[str(key)][0])
                branches_reader[str(key)][0] = integer_branches_tree[str(key)][0]
            else:
                #print '%s: %f' %(key, branches_ttree[str(key)][0] )
                branches_reader[str(key)][0] = branches_ttree[str(key)][0]

        EventWeight_ = array('d',[0])
        EventWeight_ = sample_ttree.EventWeight

        if categorise == True:
            event_classification = max(tmvareader.EvaluateMulticlass('DNN')[0],tmvareader.EvaluateMulticlass('DNN')[1],tmvareader.EvaluateMulticlass('DNN')[2])
            #print 'event_classification: ' , event_classification
            if event_classification == tmvareader.EvaluateMulticlass('DNN')[0]:
                histo_ttHclassified_events.Fill(event_classification,EventWeight_)
            elif event_classification == tmvareader.EvaluateMulticlass('DNN')[1]:
                histo_ttVclassified_events.Fill(event_classification,EventWeight_)
            elif event_classification == tmvareader.EvaluateMulticlass('DNN')[2]:
                histo_ttJclassified_events.Fill(event_classification,EventWeight_)
        else:
            #print 'event_classification: ' , event_classification
            histo_ttHclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[0],EventWeight_)
            histo_ttVclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[1],EventWeight_)
            histo_ttJclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[2],EventWeight_)

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

    # Setup TMVA
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()
    reader = TMVA.Reader("Color:!Silent")

    input_file_ttH = 'samples/2017_updated_MC/2LSS/TTH_2LSS.root'
    input_file_ttV = 'samples/2017_updated_MC/2LSS/ttV.root'
    #input_file_ttJets = 'samples/2017_updated_MC/2LSS/TTJets.root'
    input_file_ttJets = 'samples/2017_updated_MC/2LSS/Fakes_2LSS.root'

    # Check files exist
    if not isfile(input_file_ttH):
        print 'No such input file: %s' % input_file_ttH
    if not isfile(input_file_ttJets):
        print 'No such input file: %s' % input_file_ttJets
    if not isfile(input_file_ttV):
        print 'No such input file: %s' % input_file_ttV

    # Open files and load ttrees
    data_ttH = TFile.Open(input_file_ttH)
    data_ttV = TFile.Open(input_file_ttV)
    data_ttJets = TFile.Open(input_file_ttJets)
    data_ttH_tree = data_ttH.Get('syncTree')
    data_ttV_tree = data_ttV.Get('syncTree')
    data_ttJets_tree = data_ttJets.Get('syncTree')

    branches_tree = {}
    integer_branches_tree = {}
    for key, value in new_variable_list:
        #branches_tree[key] = array('d', [-999])
        branches_tree[key] = array('f', [-999])
        if 'hadTop_BDT' in key:
            keyname = 'hadTop_BDT'
            data_ttH_tree.SetBranchAddress(str(keyname), branches_tree[key])
            data_ttV_tree.SetBranchAddress(str(keyname), branches_tree[key])
            data_ttJets_tree.SetBranchAddress(str(keyname), branches_tree[key])
        elif 'Hj1_BDT' in key:
            keyname = 'Hj1_BDT'
            data_ttH_tree.SetBranchAddress(str(keyname), branches_tree[key])
            data_ttV_tree.SetBranchAddress(str(keyname), branches_tree[key])
            data_ttJets_tree.SetBranchAddress(str(keyname), branches_tree[key])
        elif ('n_fakeablesel_mu' in key) or ('n_fakeablesel_ele' in key) or ('Jet_numLoose' in key):
            integer_branches_tree[key] = array('I', [9999])
            data_ttH_tree.SetBranchAddress(str(key), integer_branches_tree[key])
            data_ttH_tree.SetBranchAddress(str(key), integer_branches_tree[key])
            data_ttJets_tree.SetBranchAddress(str(key), integer_branches_tree[key])
        else:
            data_ttH_tree.SetBranchAddress(str(key), branches_tree[key])
            data_ttV_tree.SetBranchAddress(str(key), branches_tree[key])
            data_ttJets_tree.SetBranchAddress(str(key), branches_tree[key])


    branches_reader = {}
    # Register names of inputs with reader. Together with the name give the address of the local variable that carries the updated input variables during event loop.
    for key, value in new_variable_list:
        print 'Add variable name %s: ' % key
        branches_reader[key] = array('f', [-999])
        reader.AddVariable(str(key), branches_reader[key])

    event_number = array('f',[-999])
    reader.AddSpectator('nEvent', event_number)

    num_inputs = 0
    for key, value in new_variable_list:
        num_inputs = num_inputs + 1

    classifier_suffix = opt.input_suffix

    # Book methods
    # First argument is user defined name. Doesn not have to be same as training name.
    # True type of method and full configuration are read from the weights file specified in the second argument.
    mva_weights_dir = 'MultiClass_DNN_%sVars_%s/weights/Factory_MultiClass_DNN_%sVars_%s_DNN.weights.xml' % (str(num_inputs),classifier_suffix,str(num_inputs),classifier_suffix)
    print 'using weights file: ', mva_weights_dir
    reader.BookMVA('DNN', TString(mva_weights_dir))

    classifier_samples_dir = 'MultiClass_DNN_%sVars_%s/outputs' % (str(num_inputs),classifier_suffix)
    classifier_plots_dir = 'MultiClass_DNN_%sVars_%s/plots' % (str(num_inputs),classifier_suffix)
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)
    if not os.path.exists(classifier_samples_dir):
        os.makedirs(classifier_samples_dir)

    # Define outputs: files to store histograms/ttree with results from application of classifiers and any histos/trees themselves.
    output_file_name = '%s/Applied_MultiClass_DNN_%sVars_%s.root' % (classifier_samples_dir,str(num_inputs),classifier_suffix)
    output_file = TFile.Open(output_file_name,'RECREATE')

    # Evaluate network and make plots of the response on each of the nodes to each of the simulated samples.
    #network_evaluation(data_ttH_tree, new_variable_list, 'ttH', branches_tree, branches_reader, reader, False)
    #network_evaluation(data_ttV_tree, new_variable_list, 'ttV', branches_tree, branches_reader, reader, False)
    #network_evaluation(data_ttJets_tree, new_variable_list, 'ttJets', branches_tree, branches_reader, reader, False)

    # Evaluate network and use max node response to categorise event. Only maximum node response will be plotted per event meaning each event will only contribute in the maximum nodes response histogram.
    network_evaluation(data_ttH_tree, new_variable_list, 'ttH', branches_tree, integer_branches_tree, branches_reader, reader, True)
    network_evaluation(data_ttV_tree, new_variable_list, 'ttV', branches_tree, integer_branches_tree, branches_reader, reader, True)
    network_evaluation(data_ttJets_tree, new_variable_list, 'ttJets', branches_tree, integer_branches_tree, branches_reader, reader, True)

    output_file.Close()

main()
