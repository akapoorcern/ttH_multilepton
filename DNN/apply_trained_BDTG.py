#!/usr/bin/env python
import ROOT
import os
import sys
from collections import OrderedDict
from ROOT import TMVA, TFile, TString, TLegend, THStack, TTree, TBranch
from array import array
from subprocess import call
from os.path import isfile
import optparse
import json
import math

def network_evaluation(sample_ttree, key_list, sample_name, branches_ttree, branches_reader, tmvareader):

    print 'Evaluating %s sample ' % sample_name
    print 'NEntries = ', sample_ttree.GetEntries()
    histo_sample_name = 'histo_response_%s' % sample_name
    histo_sample_title = 'BDTG Response %s sample' % sample_name
    histo_sample = ROOT.TH1D(histo_sample_name,histo_sample_title,40,-1.,1.)

    temp_percentage_done = 0
    for i in range(sample_ttree.GetEntries()):
        percentage_done = int(100*float(i)/float(sample_ttree.GetEntries()))
        if percentage_done % 10 == 0:
            if percentage_done != temp_percentage_done:
                print percentage_done
                temp_percentage_done = percentage_done
        sample_ttree.GetEntry(i)
        for reader_key, ttree_key in key_list:
            if 'hadTop_BDT' in ttree_key:
                branches_reader[str(reader_key)][0] = max(branches_ttree[str(ttree_key)][0],-1)
            elif 'Hj1_BDT' in ttree_key:
                branches_reader[str(reader_key)][0] = max(branches_ttree[str(ttree_key)][0],-1)
            else:
                branches_reader[str(reader_key)][0] = branches_ttree[str(ttree_key)][0]
        histo_sample.Fill(tmvareader.EvaluateMVA('BDTG'))
    histo_sample.Write()

def main():

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify inputs from network training',      default=None,        type='string')

    (opt, args) = parser.parse_args()

    '''if opt.json == None:
        print 'input variable .json not defined!'
        sys.exit(1)'''
    if opt.input_suffix == None:
        print 'Input files suffix not defined!'
        sys.exit(1)

    classifier_suffix = opt.input_suffix
    classifier_parent_dir = 'BinaryClassifier_BDTG_%s' % (classifier_suffix)

    # Setup TMVA
    TMVA.Tools.Instance()
    TMVA.PyMethodBase.PyInitialize()
    reader = TMVA.Reader("Color:!Silent")

    #ttH_filename = 'samples/2017MC/training_region/TTHnobb.root'
    ttH_filename = 'samples/DiLepTR_ttH_bInclude.root'
    data_ttH = TFile.Open(ttH_filename)
    #data_ttH_tree = data_ttH.Get('syncTree')
    data_ttH_tree = data_ttH.Get('BOOM')
    print 'ttH file to evaluate: ',ttH_filename

    if classifier_suffix == "ttHvsttJets":
        #bckg_filename = 'samples/2017MC/testing_region/TTJets.root'
        #bckg_filename = 'samples/2017MC/training_region/TTJets.root'
        bckg_filename = 'samples/DiLepTR_ttJets_bInclude.root'
        data_bckg = TFile.Open(bckg_filename)
        #data_bckg_tree = data_bckg.Get('syncTree')
        data_bckg_tree = data_bckg.Get('BOOM')
    elif classifier_suffix == "ttHvsttV":
        #bckg_filename = 'samples/2017MC/testing_region/ttV.root'
        #bckg_filename = 'samples/2017MC/training_region/ttV.root'
        bckg_filename = 'samples/DiLepTR_ttV_bInclude.root'
        data_bckg = TFile.Open(bckg_filename)
        #data_bckg_tree = data_bckg.Get('syncTree')
        data_bckg_tree = data_bckg.Get('BOOM')


    print 'ttJets file to evaluate: ',bckg_filename

    keys_list = []
    if classifier_suffix == "ttHvsttJets":
        keys_list = [
        ("max(abs(LepGood_eta[iLepFO_Recl[0]]),abs(LepGood_eta[iLepFO_Recl[1]]))","maxeta"),
        ("nJet25_Recl","Jet_numLoose"),
        ("mindr_lep1_jet","mindrlep1jet"),
        ("mindr_lep2_jet","mindrlep2jet"),
        ("MT_met_lep1","SR_InvarMassT"),
        ("max(-1.1,BDTv8_eventReco_mvaValue)","hadTop_BDT")
        ]
    elif classifier_suffix == "ttHvsttV":
        keys_list = [
        ("max(abs(LepGood_eta[iLepFO_Recl[0]]),abs(LepGood_eta[iLepFO_Recl[1]]))","maxeta"),
        ("nJet25_Recl","Jet_numLoose"),
        ("mindr_lep1_jet","mindrlep1jet"),
        ("mindr_lep2_jet","mindrlep2jet"),
        ("MT_met_lep1","SR_InvarMassT"),
        ("LepGood_conePt[iLepFO_Recl[1]]","corrptlep1"),
        ("LepGood_conePt[iLepFO_Recl[0]]","corrptlep2"),
        ("max(-1.1,BDTv8_eventReco_Hj_score)","Hj1_BDT")
        ]

    branches_ttree = {}
    for reader_key, ttree_key in keys_list:
        print 'Setting Branch address: ', ttree_key
        if ttree_key == 'Jet_numLoose':
            #branches_ttree[ttree_key] = array('I', [999])
            branches_ttree[ttree_key] = array('d', [999])
        else:
            #branches_ttree[ttree_key] = array('f', [-999])
            branches_ttree[ttree_key] = array('d', [-999])
        if 'hadTop_BDT' in ttree_key:
            keyname = 'hadTop_BDT'
            data_ttH_tree.SetBranchAddress(str(keyname), branches_ttree[ttree_key])
            data_bckg_tree.SetBranchAddress(str(keyname), branches_ttree[ttree_key])
        elif 'Hj1_BDT' in ttree_key:
            keyname = 'Hj1_BDT'
            data_ttH_tree.SetBranchAddress(str(keyname), branches_ttree[ttree_key])
            data_bckg_tree.SetBranchAddress(str(keyname), branches_ttree[ttree_key])
        else:
            data_ttH_tree.SetBranchAddress(str(ttree_key), branches_ttree[ttree_key])
            data_bckg_tree.SetBranchAddress(str(ttree_key), branches_ttree[ttree_key])

    # Keep track of event numbers for cross-checks.
    event_number_branch = array('L',[999])
    data_ttH_tree.SetBranchAddress('nEvent', event_number_branch)

    branches_reader = {}
    # Register names of inputs with reader. Together with the name give the address of the local variable that carries the updated input variables during event loop.
    for reader_key, ttree_key in keys_list:
        print 'Add variable name %s: ' % reader_key
        branches_reader[reader_key] = array('f', [-999])
        reader.AddVariable(str(reader_key), branches_reader[reader_key])

    iLepFO_Recl0 = array('f',[-999])
    reader.AddSpectator('iLepFO_Recl[0]', iLepFO_Recl0)
    iLepFO_Recl1 = array('f',[-999])
    reader.AddSpectator('iLepFO_Recl[1]', iLepFO_Recl1)
    iLepFO_Recl2 = array('f',[-999])
    reader.AddSpectator('iLepFO_Recl[2]', iLepFO_Recl2)

    # Book methods
    # First argument is user defined name. Doesn not have to be same as training name.
    # True type of method and full configuration are read from the weights file specified in the second argument.
    if classifier_suffix == "ttHvsttJets":
        mva_weights_dir = 'BinaryClassifier_BDTG_%s/weights/2lss_ttbar_withBDTv8_BDTG.weights.xml' % classifier_suffix
    elif classifier_suffix == "ttHvsttV":
        mva_weights_dir = 'BinaryClassifier_BDTG_%s/weights/2lss_ttV_withHj_BDTG.weights.xml' % classifier_suffix

    print 'using weights file: ', mva_weights_dir
    reader.BookMVA('BDTG', TString(mva_weights_dir))

    classifier_samples_dir = classifier_parent_dir+"/outputs"
    classifier_plots_dir = classifier_parent_dir+"/plots"
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)
    if not os.path.exists(classifier_samples_dir):
        os.makedirs(classifier_samples_dir)

    # Define outputs: files to store histograms/ttree with results from application of classifiers and any histos/trees themselves.
    output_file_name = '%s/Applied_%s.root' % (classifier_samples_dir,classifier_parent_dir)
    output_file = TFile.Open(output_file_name,'RECREATE')


    network_evaluation(data_ttH_tree, keys_list, 'ttH', branches_ttree, branches_reader, reader)
    if classifier_suffix == 'ttHvsttJets':
        network_evaluation(data_bckg_tree, keys_list, 'ttjets', branches_ttree, branches_reader, reader)
    if classifier_suffix == 'ttHvsttV':
        network_evaluation(data_bckg_tree, keys_list, 'ttV', branches_ttree, branches_reader, reader)
    output_file.Close()

main()
