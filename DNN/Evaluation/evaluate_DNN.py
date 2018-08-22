#!/usr/bin/env python
# Select Theano as backend for Keras
import ROOT
import os
import sys
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
# Set architecture of system (AVX instruction set is not supported on SWAN)
#environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'
from ROOT import TMVA, TFile, TString, TLegend, THStack, TTree, TBranch, gDirectory
from array import array
from subprocess import call
from os.path import isfile
import optparse
import json
import math

def network_evaluation(sample_ttree, variables_list, sample_name, branches_ttree, integer_branches_tree, branches_reader, tmvareader, categorise, output_tree):

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

    eval_ttHnode = array('f',[0.])
    eval_ttVnode = array('f',[0.])
    eval_ttJnode = array('f',[0.])
    ttH_branch = output_tree.Branch('DNN_ttHnode', eval_ttHnode, 'DNN_ttHnode/F')
    ttV_branch = output_tree.Branch('DNN_ttVnode', eval_ttVnode, 'DNN_ttVnode/F')
    ttJ_branch = output_tree.Branch('DNN_ttJnode', eval_ttJnode, 'DNN_ttJnode/F')

    temp_percentage_done = 0
    for i in range(sample_ttree.GetEntries()):
    #for i in range(1000):
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


        # Sanity Check: Ensure variables going into evaluation make sense.
        '''print '>>>>>> Event #: %i <<<<<<<' % sample_ttree.nEvent
        for key, value in variables_list:
            print '%s: %f' %(key, branches_reader[str(key)][0] )'''

        EventWeight_ = array('d',[0])
        EventWeight_ = sample_ttree.EventWeight

        if categorise == True:
            event_classification = max(tmvareader.EvaluateMulticlass('DNN')[0],tmvareader.EvaluateMulticlass('DNN')[1],tmvareader.EvaluateMulticlass('DNN')[2])
            #print 'event_classification = ' , event_classification
            if event_classification == tmvareader.EvaluateMulticlass('DNN')[0]:
                #print 'ttH category'
                histo_ttHclassified_events.Fill(event_classification,EventWeight_)
                eval_ttHnode[0] = event_classification
                eval_ttVnode[0] = -999
                eval_ttJnode[0] = -999
            elif event_classification == tmvareader.EvaluateMulticlass('DNN')[1]:
                #print 'ttV category'
                histo_ttVclassified_events.Fill(event_classification,EventWeight_)
                eval_ttHnode[0] = -999
                eval_ttVnode[0] = event_classification
                eval_ttJnode[0] = -999
            elif event_classification == tmvareader.EvaluateMulticlass('DNN')[2]:
                #print 'ttJ category'
                histo_ttJclassified_events.Fill(event_classification,EventWeight_)
                eval_ttHnode[0] = -999
                eval_ttVnode[0] = -999
                eval_ttJnode[0] = event_classification
        else:
            histo_ttHclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[0],EventWeight_)
            histo_ttVclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[1],EventWeight_)
            histo_ttJclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[2],EventWeight_)
            eval_ttHnode[0] = tmvareader.EvaluateMulticlass('DNN')[0]
            eval_ttVnode[0] = tmvareader.EvaluateMulticlass('DNN')[1]
            eval_ttJnode[0] = tmvareader.EvaluateMulticlass('DNN')[2]

        ttH_branch.Fill()
        ttV_branch.Fill()
        ttJ_branch.Fill()

    histo_ttHclassified_events.Write()
    histo_ttVclassified_events.Write()
    histo_ttJclassified_events.Write()

def main():

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify inputs from network training',      default=None,        type='string')
    parser.add_option('-j', '--json',        dest='json'  ,      help='json file with list of variables',      default=None,        type='string'),
    parser.add_option('-i', '--input',        dest='input_file'  ,      help='input file',      default=None,        type='string')

    (opt, args) = parser.parse_args()
    jsonFile = open(opt.json,'r')

    if opt.json == None:
        print 'input variable .json not defined!'
        sys.exit(1)
    if opt.input_suffix == None:
        print 'Input files suffix not defined!'
        sys.exit(1)
    if opt.input_file == None:
        print 'Input file not defined!'
        sys.exit(1)

    new_variable_list = json.load(jsonFile,encoding="utf-8").items()
    n_input_vars = 0
    for key, value in new_variable_list:
        n_input_vars = n_input_vars + 1

    input_file = opt.input_file

    classifier_suffix = opt.input_suffix
    classifier_parent_dir = '/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/Evaluation/MultiClass_DNN_%sVars_%s' % (str(n_input_vars),classifier_suffix,)

    # Setup TMVA
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()
    reader = TMVA.Reader("Color:!Silent")

    # Check files exist
    if not isfile(input_file):
        print 'No such input file: %s' % input_file

    # Open files and load ttrees
    data_file = TFile.Open(input_file)
    data_tree = data_file.Get("syncTree")

    branches_tree = {}
    integer_branches_tree = {}
    for key, value in new_variable_list:
        if 'hadTop_BDT' in key:
            branches_tree[key] = array('f', [-999])
            keyname = 'hadTop_BDT'
            data_tree.SetBranchAddress(str(keyname), branches_tree[key])
        elif 'Hj1_BDT' in key:
            branches_tree[key] = array('f', [-999])
            keyname = 'Hj1_BDT'
            data_tree.SetBranchAddress(str(keyname), branches_tree[key])
        elif ('n_fakeablesel_mu' in key) or ('n_fakeablesel_ele' in key) or ('Jet_numLoose' in key):
            integer_branches_tree[key] = array('I', [9999])
            data_tree.SetBranchAddress(str(key), integer_branches_tree[key])
        else:
            branches_tree[key] = array('f', [-999])
            data_tree.SetBranchAddress(str(key), branches_tree[key])

    branches_reader = {}
    # Register names of inputs with reader. Together with the name give the address of the local variable that carries the updated input variables during event loop.
    for key, value in new_variable_list:
        print 'Add variable name %s: ' % key
        branches_reader[key] = array('f', [-999])
        reader.AddVariable(str(key), branches_reader[key])

    event_number = array('f',[-999])

    reader.AddSpectator('nEvent', event_number)

    # Book methods
    # First argument is user defined name. Doesn not have to be same as training name.
    # True type of method and full configuration are read from the weights file specified in the second argument.
    mva_weights_dir = '%s/weights/Factory_MultiClass_DNN_%sVars_%s_DNN.weights.xml' % (classifier_parent_dir,str(n_input_vars),classifier_suffix)
    print 'using weights file: ', mva_weights_dir
    reader.BookMVA('DNN', TString(mva_weights_dir))

    if '/2LSS/' in input_file:
            analysis_region = '2LSS'
    elif '/ttWctrl/' in input_file:
        analysis_region = 'ttWctrl'
    elif '/ttZctrl/' in input_file:
        analysis_region = 'ttZctrl'
    elif '/Clos/' in input_file:
        analysis_region = 'Closure'
    elif '/JESDown2L/' in input_file:
        analysis_region = 'JESDown2L'
    elif '/JESUp2L/' in input_file:
        analysis_region = 'JESUp2L'


    classifier_samples_dir = classifier_parent_dir+"/outputs"
    classifier_plots_dir = classifier_parent_dir+"/plots"
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)
    if not os.path.exists(classifier_samples_dir):
        os.makedirs(classifier_samples_dir)

    analysis_region_samples_dir = '%s/%s' % (classifier_samples_dir, analysis_region)

    analysis_region_plots_dir = '%s/%s' % (classifier_plots_dir, analysis_region)
    if not os.path.exists(analysis_region_plots_dir):
        os.makedirs(analysis_region_plots_dir)
    if not os.path.exists(analysis_region_samples_dir):
        os.makedirs(analysis_region_samples_dir)

    output_suffix = input_file[input_file.rindex('/')+1:]
    print 'output_suffix: ', output_suffix
    # Define outputs: files to store histograms/ttree with results from application of classifiers and any histos/trees themselves.
    #output_file_name = '%s/Evaluated_%s_%s' % (classifier_samples_dir,classifier_suffix,output_suffix)
    output_file_name = '%s/Evaluated_%s_%s' % (analysis_region_samples_dir,classifier_suffix,output_suffix)
    output_file = TFile.Open(output_file_name,'RECREATE')
    output_tree = data_tree.CopyTree("")
    output_tree.SetName("output_tree")
    nEvents_check = output_tree.BuildIndex("nEvent","run")
    print 'Copied %s events from original tree' % (nEvents_check)

    sample_nickname = ''

    if 'H_2LSS' in output_suffix:
        if 'TTH_2LSS' in output_suffix:
            sample_nickname = 'TTH_2LSS'
        else:
            sample_nickname = 'H_2LSS'
    if 'TTWW_2LSS' in output_suffix:
        sample_nickname = 'TTWW_2LSS'
    if 'TTW_2LSS' in output_suffix:
        sample_nickname = 'TTW_2LSS'
    if 'TTZ_2LSS' in output_suffix:
        sample_nickname = 'TTZ_2LSS'
    if 'Conv_2LSS' in output_suffix:
        sample_nickname = 'Conv_2LSS'
    if 'EWK_2LSS' in output_suffix:
        sample_nickname = 'EWK_2LSS'
    if 'Fakes_2LSS' in output_suffix:
        sample_nickname = 'Fakes2LSS'
    if 'Flips_2LSS' in output_suffix:
        sample_nickname = 'Flips_2LSS'
    if 'Rares_2LSS' in output_suffix:
        sample_nickname = 'Rares_2LSS'
    if 'TT_Clos' in output_suffix:
        sample_nickname = 'TT_Clos'
    if 'Data' in output_suffix:
        sample_nickname = 'Data'

    # Evaluate network and use max node response to categorise event. Only maximum node response will be plotted per event meaning each event will only contribute in the maximum nodes response histogram.
    network_evaluation(data_tree, new_variable_list, sample_nickname, branches_tree, integer_branches_tree, branches_reader, reader, True, output_tree)

    output_file.Write()
    gDirectory.Delete("syncTree;*")
    output_file.Close()
    print 'Job complete. Exiting.'
    sys.exit(0)

main()
