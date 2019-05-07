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
import pytz
from datetime import datetime

def network_evaluation(sample_ttree, variables_list, sample_name, branches_ttree, integer_branches_tree, branches_reader, tmvareader, categorise, output_tree):

    print 'Evaluating %s sample ' % sample_name

    if categorise == True:
        histoname_type = 'Category'
    else:
        histoname_type = 'Node'

    histo_ee_events_name = 'histo_ee_events_%s' % (sample_name)

    histo_em_ttHclassified_events_name = 'histo_em_ttH%s_events_%s' % (histoname_type,sample_name)
    histo_em_ttVclassified_events_name = 'histo_em_ttV%s_events_%s' % (histoname_type,sample_name)
    histo_em_ttJclassified_events_name = 'histo_em_ttJ%s_events_%s' % (histoname_type,sample_name)

    histo_mm_ttHclassified_events_name = 'histo_mm_ttH%s_events_%s' % (histoname_type,sample_name)
    histo_mm_ttVclassified_events_name = 'histo_mm_ttV%s_events_%s' % (histoname_type,sample_name)
    histo_mm_ttJclassified_events_name = 'histo_mm_ttJ%s_events_%s' % (histoname_type,sample_name)

    histo_ee_events_title = 'ttH ee Events: %s Sample' % (sample_name)

    histo_em_ttHclassified_events_title = 'ttH em %s Events: %s Sample' % (histoname_type,sample_name)
    histo_em_ttVclassified_events_title = 'ttV em %s Events: %s Sample' % (histoname_type,sample_name)
    histo_em_ttJclassified_events_title = 'ttJ em %s Events: %s Sample' % (histoname_type,sample_name)

    histo_mm_ttHclassified_events_title = 'ttH mm %s Events: %s Sample' % (histoname_type,sample_name)
    histo_mm_ttVclassified_events_title = 'ttV mm %s Events: %s Sample' % (histoname_type,sample_name)
    histo_mm_ttJclassified_events_title = 'ttJ mm %s Events: %s Sample' % (histoname_type,sample_name)

    histo_ee_events = ROOT.TH1D(histo_ee_events_name,histo_ee_events_title,50,0,1.)

    histo_em_ttHclassified_events = ROOT.TH1D(histo_em_ttHclassified_events_name,histo_em_ttHclassified_events_title,50,0,1.)
    histo_em_ttVclassified_events = ROOT.TH1D(histo_em_ttVclassified_events_name,histo_em_ttVclassified_events_title,50,0,1.)
    histo_em_ttJclassified_events = ROOT.TH1D(histo_em_ttJclassified_events_name,histo_em_ttJclassified_events_title,50,0,1.)

    histo_mm_ttHclassified_events = ROOT.TH1D(histo_mm_ttHclassified_events_name,histo_mm_ttHclassified_events_title,50,0,1.)
    histo_mm_ttVclassified_events = ROOT.TH1D(histo_mm_ttVclassified_events_name,histo_mm_ttVclassified_events_title,50,0,1.)
    histo_mm_ttJclassified_events = ROOT.TH1D(histo_mm_ttJclassified_events_name,histo_mm_ttJclassified_events_title,50,0,1.)

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

            lep_cat = ''

            if (sample_ttree.SubCat2l == 1) or (sample_ttree.SubCat2l == 2):
                lep_cat = 'ee'
            if (sample_ttree.SubCat2l == 3) or (sample_ttree.SubCat2l == 4) or (sample_ttree.SubCat2l == 5) or (sample_ttree.SubCat2l == 6):
                lep_cat = 'em'
            if (sample_ttree.SubCat2l == 7) or (sample_ttree.SubCat2l == 8) or (sample_ttree.SubCat2l == 9) or (sample_ttree.SubCat2l == 10):
                lep_cat = 'mm'

            event_classification = max(tmvareader.EvaluateMulticlass('DNN')[0],tmvareader.EvaluateMulticlass('DNN')[1],tmvareader.EvaluateMulticlass('DNN')[2])
            if lep_cat == 'ee':
                histo_ee_events.Fill(event_classification,EventWeight_)

            if event_classification == tmvareader.EvaluateMulticlass('DNN')[0]:
                if lep_cat == 'em':
                    histo_em_ttHclassified_events.Fill(event_classification,EventWeight_)
                if lep_cat == 'mm':
                    histo_mm_ttHclassified_events.Fill(event_classification,EventWeight_)
                eval_ttHnode[0] = event_classification
                eval_ttVnode[0] = -999
                eval_ttJnode[0] = -999
            elif event_classification == tmvareader.EvaluateMulticlass('DNN')[1]:
                if lep_cat == 'em':
                    histo_em_ttVclassified_events.Fill(event_classification,EventWeight_)
                if lep_cat == 'mm':
                    histo_mm_ttVclassified_events.Fill(event_classification,EventWeight_)
                eval_ttHnode[0] = -999
                eval_ttVnode[0] = event_classification
                eval_ttJnode[0] = -999
            elif event_classification == tmvareader.EvaluateMulticlass('DNN')[2]:
                if lep_cat == 'em':
                    histo_em_ttJclassified_events.Fill(event_classification,EventWeight_)
                if lep_cat == 'mm':
                    histo_mm_ttJclassified_events.Fill(event_classification,EventWeight_)
                eval_ttHnode[0] = -999
                eval_ttVnode[0] = -999
                eval_ttJnode[0] = event_classification
        else:
            if lep_cat == 'ee':
                histo_ee_events.Fill(tmvareader.EvaluateMulticlass('DNN')[2],EventWeight_)
                histo_ee_events.Fill(tmvareader.EvaluateMulticlass('DNN')[1],EventWeight_)
                histo_ee_events.Fill(tmvareader.EvaluateMulticlass('DNN')[0],EventWeight_)
            if lep_cat == 'em':
                histo_em_ttJclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[2],EventWeight_)
                histo_em_ttVclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[1],EventWeight_)
                histo_em_ttHclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[0],EventWeight_)
            if lep_cat == 'mm':
                histo_mm_ttJclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[2],EventWeight_)
                histo_mm_ttVclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[1],EventWeight_)
                histo_mm_ttHclassified_events.Fill(tmvareader.EvaluateMulticlass('DNN')[0],EventWeight_)
            eval_ttHnode[0] = tmvareader.EvaluateMulticlass('DNN')[0]
            eval_ttVnode[0] = tmvareader.EvaluateMulticlass('DNN')[1]
            eval_ttJnode[0] = tmvareader.EvaluateMulticlass('DNN')[2]

        ttH_branch.Fill()
        ttV_branch.Fill()
        ttJ_branch.Fill()

    histo_ee_events.Write()

    histo_em_ttJclassified_events.Write()
    histo_em_ttVclassified_events.Write()
    histo_em_ttHclassified_events.Write()

    histo_mm_ttJclassified_events.Write()
    histo_mm_ttVclassified_events.Write()
    histo_mm_ttHclassified_events.Write()

def main():

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify inputs from network training',      default=None,        type='string')
    parser.add_option('-j', '--json',        dest='json'  ,      help='json file with list of variables',      default=None,        type='string')
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
    classifier_parent_dir = '/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/Evaluation/V7-DNN_%s' % (classifier_suffix)

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
    mva_weights_dir = '%s/weights/Factory_V7-DNN_%s_DNN.weights.xml' % (classifier_parent_dir,classifier_suffix)
    print 'using weights file: ', mva_weights_dir
    reader.BookMVA('DNN', TString(mva_weights_dir))

    if '/2L/' in input_file:
            analysis_region = '2L'
    elif '/ttWctrl/' in input_file:
        analysis_region = 'ttWctrl'
    elif '/JESDownttWctrl/' in input_file:
        analysis_region = 'JESDownttWctrl'
    elif '/JESUpttWctrl/' in input_file:
        analysis_region = 'JESUpttWctrl'
    elif '/ClosTTWctrl/' in input_file:
        analysis_region = 'ClosTTWctrl'
    elif '/ttZctrl/' in input_file:
        analysis_region = 'ttZctrl'
    elif '/Clos2LSS/' in input_file:
        analysis_region = 'Closure'
    elif '/JESDown2L/' in input_file:
        analysis_region = 'JESDown2L'
    elif '/JESUp2L/' in input_file:
        analysis_region = 'JESUp2L'

    time_suffix = str(datetime.now(pytz.utc)).split(' ')
    print time_suffix[0]
    #classifier_samples_dir = classifier_parent_dir+"/outputs"
    classifier_samples_dir = classifier_parent_dir+"/outputs-newbinning"
    #classifier_plots_dir = classifier_parent_dir+"/plots"
    classifier_plots_dir = classifier_parent_dir+"/plots-newbinning"
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
    output_file_name = '%s/Evaluated_%s_%s' % (analysis_region_samples_dir,classifier_suffix,output_suffix)
    output_file = TFile.Open(output_file_name,'RECREATE')
    output_tree = data_tree.CopyTree("")
    output_tree.SetName("output_tree")
    nEvents_check = output_tree.BuildIndex("nEvent","run")
    print 'Copied %s events from original tree' % (nEvents_check)

    sample_nickname = ''

    if 'THQ_htt_2L' in input_file:
        sample_nickname = 'THQ_htt_2L'
    if 'THQ_hzz_2L' in input_file:
        sample_nickname = 'THQ_hzz_2L'
    if 'THW_hww_2L' in input_file:
        sample_nickname = 'THW_hww_2L'
    if 'TTH_hmm_2L' in input_file:
        sample_nickname = 'TTH_hmm_2L'
    if 'TTH_htt_2L' in input_file:
        sample_nickname = 'TTH_htt_2L'
    if 'TTH_hzz_2L' in input_file:
        sample_nickname = 'TTH_hzz_2L'
    if 'THQ_hww_2L' in input_file:
        sample_nickname = 'THQ_hww_2L'
    if 'THW_htt_2L' in input_file:
        sample_nickname = 'THW_htt_2L'
    if 'THW_hzz_2L' in input_file:
        sample_nickname = 'THW_hzz_2L'
    if 'TTH_hot_2L' in input_file:
        sample_nickname = 'TTH_hot_2L'
    if 'TTH_hww_2L' in input_file:
        sample_nickname = 'TTH_hww_2L'
    if 'TTWW_2L' in input_file:
        sample_nickname = 'TTWW_2L'
    if 'TTW_2L' in input_file:
        sample_nickname = 'TTW_2L'
    if 'TTZ_2L' in input_file:
        sample_nickname = 'TTZ_2L'
    if 'Conv_2L' in input_file:
        sample_nickname = 'Conv_2L'
    if 'EWK_2L' in input_file:
        sample_nickname = 'EWK_2L'
    if 'Fakes_2L' in input_file:
        sample_nickname = 'Fakes_2L'
    if 'Flips_2L' in input_file:
        sample_nickname = 'Flips_2L'
    if 'Rares_2L' in input_file:
        sample_nickname = 'Rares_2L'
    if 'TT_Clos' in input_file:
        sample_nickname = 'TT_Clos'
    if 'Data' in input_file:
        sample_nickname = 'Data'

    # Evaluate network and use max node response to categorise event. Only maximum node response will be plotted per event meaning each event will only contribute in the maximum nodes response histogram.
    network_evaluation(data_tree, new_variable_list, sample_nickname, branches_tree, integer_branches_tree, branches_reader, reader, True, output_tree)

    output_file.Write()
    gDirectory.Delete("syncTree;*")
    output_file.Close()
    print 'Job complete. Exiting.'
    sys.exit(0)

main()
