#!/usr/bin/env python
######################################
# shape_template_creator.py
# Joshuha Thomas-Wilsker
# IHEP, CAS, CERN
######################################
# Code to create .root files for
# CMS combined tool. Each file
# represents an analysis bin. Inside
# the files one will find a histogram
# of the discromonant distribution
# per physics process for the noominal
# and systematic variations along with
# the histogram for the observed data.
######################################

import ROOT
import os
import sys
import math
#from os import environ
#environ['KERAS_BACKEND'] = 'tensorflow'
from ROOT import TFile, TString, TH1D, TH1F, TMVA
from array import array
from subprocess import call
from os.path import isfile
import optparse
import json


def fill_closure_histos(sample_name,analysis_bin,closure_histos_m_TT,closure_histos_m_QCD,closure_histos_e_TT,closure_histos_e_QCD):
    print 'fill_closure_histos'
    closure_file = TFile.Open("/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/Evaluation/MultiClass_DNN_17Vars_2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs/outputs/Closure/Evaluated_2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs_TT_Clos.root")
    closure_tree = closure_file.Get("output_tree")

    # SetBranchAddress for variables used in discriminant
    closure_variables_json_name = '/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/LHFit/templates/closure_variable_list.json'
    closure_variables_json = open(closure_variables_json_name,'r')
    if closure_variables_json == None:
        print 'No closure variables file'
        sys.exit(1)
    closure_variables = json.load(closure_variables_json,encoding='utf-8').items()

    closure_branches = {}

    # SetBranchAddress for each variable used in closure test.
    closure_branches['DNN_ttHnode'] = array('f',[-999])
    closure_branches['DNN_ttVnode'] = array('f',[-999])
    closure_branches['DNN_ttJnode'] = array('f',[-999])
    closure_branches['closWeight'] = array('f',[-999])
    closure_branches['FakeRate'] = array('f',[-999])
    closure_branches['FakeRate_m_TT'] = array('f',[-999])
    closure_branches['FakeRate_m_QCD'] = array('f',[-999])
    closure_branches['FakeRate_e_TT'] = array('f',[-999])
    closure_branches['FakeRate_e_QCD'] = array('f',[-999])

    closure_tree.SetBranchAddress('EventWeight', closure_branches['closWeight'])
    closure_tree.SetBranchAddress('FakeRate', closure_branches['FakeRate'])
    closure_tree.SetBranchAddress('FakeRate_m_TT', closure_branches['FakeRate_m_TT'])
    closure_tree.SetBranchAddress('FakeRate_m_QCD', closure_branches['FakeRate_m_QCD'])
    closure_tree.SetBranchAddress('FakeRate_e_TT', closure_branches['FakeRate_e_TT'])
    closure_tree.SetBranchAddress('FakeRate_e_QCD', closure_branches['FakeRate_e_QCD'])
    closure_tree.SetBranchAddress('DNN_ttHnode',closure_branches['DNN_ttHnode'])
    closure_tree.SetBranchAddress('DNN_ttVnode',closure_branches['DNN_ttVnode'])
    closure_tree.SetBranchAddress('DNN_ttJnode',closure_branches['DNN_ttJnode'])

    # Event loop
    for event in range(closure_tree.GetEntries()):
        closure_tree.GetEntry(event)

        if '_ee' in analysis_bin:
            if (closure_tree.SubCat2l != 1) and (closure_tree.SubCat2l != 2):
                continue
        if '_em' in analysis_bin:
            if (closure_tree.SubCat2l != 3) and (closure_tree.SubCat2l != 4) and (closure_tree.SubCat2l != 5) and (closure_tree.SubCat2l != 6):
                continue
        if '_mm' in analysis_bin:
            if (closure_tree.SubCat2l != 7) and (closure_tree.SubCat2l != 8) and (closure_tree.SubCat2l != 9) and (closure_tree.SubCat2l != 10):
                continue

        # Fill closure histograms
        if closure_tree.EventWeight!=0:
            if '_em' in analysis_bin or '_mm' in analysis_bin:
                if 'ttHCat' in analysis_bin and closure_tree.DNN_ttHnode != -999:
                    closure_histos_m_QCD.Fill(closure_tree.DNN_ttHnode,closure_tree.EventWeight*closure_tree.FakeRate_m_QCD/closure_tree.FakeRate)
                    closure_histos_m_TT.Fill(closure_tree.DNN_ttHnode,closure_tree.EventWeight*closure_tree.FakeRate_m_TT/closure_tree.FakeRate)
                    closure_histos_e_QCD.Fill(closure_tree.DNN_ttHnode,closure_tree.EventWeight*closure_tree.FakeRate_e_QCD/closure_tree.FakeRate)
                    closure_histos_e_TT.Fill(closure_tree.DNN_ttHnode,closure_tree.EventWeight*closure_tree.FakeRate_e_TT/closure_tree.FakeRate)
                if 'ttVCat' in analysis_bin and closure_tree.DNN_ttVnode != -999:
                    closure_histos_m_QCD.Fill(closure_tree.DNN_ttVnode,closure_tree.EventWeight*closure_tree.FakeRate_m_QCD/closure_tree.FakeRate)
                    closure_histos_m_TT.Fill(closure_tree.DNN_ttVnode,closure_tree.EventWeight*closure_tree.FakeRate_m_TT/closure_tree.FakeRate)
                    closure_histos_e_QCD.Fill(closure_tree.DNN_ttVnode,closure_tree.EventWeight*closure_tree.FakeRate_e_QCD/closure_tree.FakeRate)
                    closure_histos_e_TT.Fill(closure_tree.DNN_ttVnode,closure_tree.EventWeight*closure_tree.FakeRate_e_TT/closure_tree.FakeRate)
                if 'ttJCat' in analysis_bin and closure_tree.DNN_ttJnode != -999:
                    closure_histos_m_QCD.Fill(closure_tree.DNN_ttJnode,closure_tree.EventWeight*closure_tree.FakeRate_m_QCD/closure_tree.FakeRate)
                    closure_histos_m_TT.Fill(closure_tree.DNN_ttJnode,closure_tree.EventWeight*closure_tree.FakeRate_m_TT/closure_tree.FakeRate)
                    closure_histos_e_QCD.Fill(closure_tree.DNN_ttJnode,closure_tree.EventWeight*closure_tree.FakeRate_e_QCD/closure_tree.FakeRate)
                    closure_histos_e_TT.Fill(closure_tree.DNN_ttJnode,closure_tree.EventWeight*closure_tree.FakeRate_e_TT/closure_tree.FakeRate)
            if '_ee' in analysis_bin:
                if closure_tree.DNN_ttHnode != -999:
                    closure_histos_m_QCD.Fill(closure_tree.DNN_ttHnode,closure_tree.EventWeight*closure_tree.FakeRate_m_QCD/closure_tree.FakeRate)
                    closure_histos_m_TT.Fill(closure_tree.DNN_ttHnode,closure_tree.EventWeight*closure_tree.FakeRate_m_TT/closure_tree.FakeRate)
                    closure_histos_e_QCD.Fill(closure_tree.DNN_ttHnode,closure_tree.EventWeight*closure_tree.FakeRate_e_QCD/closure_tree.FakeRate)
                    closure_histos_e_TT.Fill(closure_tree.DNN_ttHnode,closure_tree.EventWeight*closure_tree.FakeRate_e_TT/closure_tree.FakeRate)
                if closure_tree.DNN_ttVnode != -999:
                    closure_histos_m_QCD.Fill(closure_tree.DNN_ttVnode,closure_tree.EventWeight*closure_tree.FakeRate_m_QCD/closure_tree.FakeRate)
                    closure_histos_m_TT.Fill(closure_tree.DNN_ttVnode,closure_tree.EventWeight*closure_tree.FakeRate_m_TT/closure_tree.FakeRate)
                    closure_histos_e_QCD.Fill(closure_tree.DNN_ttVnode,closure_tree.EventWeight*closure_tree.FakeRate_e_QCD/closure_tree.FakeRate)
                    closure_histos_e_TT.Fill(closure_tree.DNN_ttVnode,closure_tree.EventWeight*closure_tree.FakeRate_e_TT/closure_tree.FakeRate)
                if closure_tree.DNN_ttJnode != -999:
                    closure_histos_m_QCD.Fill(closure_tree.DNN_ttJnode,closure_tree.EventWeight*closure_tree.FakeRate_m_QCD/closure_tree.FakeRate)
                    closure_histos_m_TT.Fill(closure_tree.DNN_ttJnode,closure_tree.EventWeight*closure_tree.FakeRate_m_TT/closure_tree.FakeRate)
                    closure_histos_e_QCD.Fill(closure_tree.DNN_ttJnode,closure_tree.EventWeight*closure_tree.FakeRate_e_QCD/closure_tree.FakeRate)
                    closure_histos_e_TT.Fill(closure_tree.DNN_ttJnode,closure_tree.EventWeight*closure_tree.FakeRate_e_TT/closure_tree.FakeRate)

def fill_stat_unc_hists(output_file, nominal_histo_, process_name, analysis_bin):
    output_file.cd()

    print 'fill_stat_unc_hists : analysis_bin: ', analysis_bin

    for bin in range(nominal_histo_.GetNbinsX()):

        tempDNN_Up_stat = nominal_histo_.Clone()
        tempDNN_Up_stat_name = '%s_%s_%s_statbin%sUp' % (process_name,process_name,analysis_bin,bin)
        tempDNN_Up_stat.SetName(tempDNN_Up_stat_name)
        tempDNN_Up_stat.Sumw2()

        tempDNN_Down_stat = nominal_histo_.Clone()
        tempDNN_Down_stat_name = '%s_%s_%s_statbin%sDown' %(process_name,process_name,analysis_bin,bin)
        tempDNN_Down_stat.SetName(tempDNN_Down_stat_name)
        tempDNN_Down_stat.Sumw2()

        tempDNN_bin_val = nominal_histo_.GetBinContent(bin)
        statuncert = tempDNN_Up_stat.GetBinError(bin)
        tempDNN_Up_stat.SetBinContent(bin, tempDNN_bin_val+statuncert)
        tempDNN_Down_stat.SetBinContent(bin, tempDNN_bin_val-statuncert)

        tempDNN_Up_stat.Write()
        tempDNN_Down_stat.Write()
        tempDNN_Up_stat.SetDirectory(0)
        tempDNN_Down_stat.SetDirectory(0)

def main():
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-j', '--json',        dest='json'  ,      help='json file with list of nominal samples',      default=None,        type='string'),
    parser.add_option('-o', '--output',        dest='out_dir'  ,      help='Name your output directory in case you need to identify the templates later.',      default=None,        type='string')
    parser.add_option('-b', '--bin',        dest='analysis_bin'  ,      help='Create template for this analysis bin',      default=None,        type='string')

    (opt, args) = parser.parse_args()

    jsonFile = open(opt.json,'r')
    if opt.json == None:
        print 'input variable .json not defined!'
        sys.exit(1)
    input_files_list = json.load(jsonFile,encoding='utf-8').items()

    # Currently have separate files with JES SF's varied to systematic uncertainty applied
    JES_up_json = open('/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/LHFit/templates/JES_up_samples_list.json','r')
    JES_up_files_list = json.load(JES_up_json,encoding='utf-8').items()
    JES_down_json = open('/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/LHFit/templates/JES_down_samples_list.json','r')
    JES_down_files_list = json.load(JES_down_json,encoding='utf-8').items()
    analysis_bin = opt.analysis_bin

    variation = ['Up','Down']

    #for analysis_bin in analysis_bin_list:
    print 'Analysis bin: ', analysis_bin

    # We want one file per category.
    output_directory_name = opt.out_dir
    output_full_path = output_directory_name
    if not os.path.exists(output_full_path):
        print 'Make directory: ', output_full_path
        os.makedirs(output_full_path)
    output_file_name = '%s/%s.input.root' % (output_full_path,analysis_bin)
    output_file = TFile.Open(output_file_name,'RECREATE')
    print 'Output file: ', output_file_name

    # Save metadata including integral of histogram used for rates in datacards.
    output_metadata_name = "%s/metadata_%s.txt" % (output_full_path,analysis_bin)
    output_metadata_file = open(output_metadata_name,"w")

    # Check and get input file
    for process_name,input_file in input_files_list:
        print 'process_name: ', process_name
        print 'input_file: ' , input_file
        if not isfile(input_file):
            print 'No such input file: %s' % input_file

        # Open file and load ttree
        data_file = TFile.Open(input_file)
        data_tree = data_file.Get("output_tree")


        syst_list = ['']
        if 'Data' == process_name:
            syst_list = ['']
        else:
            syst_list = ['JES',
                        'PU',
                        'Trig',
                        'genWeight_muF',
                        'genWeight_muR',
                        'elelooseSF',
                        'eletightSF',
                        'mulooseSF',
                        'mutightSF',
                        'bWeight_cferr1',
                        'bWeight_cferr2',
                        'bWeight_hfstats1',
                        'bWeight_hfstats2',
                        'bWeight_jes',
                        'bWeight_lf'
                         ]
        # If file contains flips/fakes we need to include additional systematics.
        if 'flips' in process_name:
            syst_list.append('ChargeMis')
        if 'fake' in process_name:
            syst_list.append('FakeRate_m_norm')
            syst_list.append('FakeRate_m_pt')
            syst_list.append('FakeRate_m_be')
            syst_list.append('FakeRate_e_norm')
            syst_list.append('FakeRate_e_pt')
            syst_list.append('FakeRate_e_be')


        # Create the nominal histograms.
        nominal_histo_name = '%s' %(process_name)
        print 'nominal_histo_name: ', nominal_histo_name
        nominal_histo_ = ROOT.TH1D(nominal_histo_name,nominal_histo_name, 10, 0, 1)
        nominal_histo_.Sumw2()

        for i in range(data_tree.GetEntries()):
            data_tree.GetEntry(i)

            if '_ee' in analysis_bin:
                if (data_tree.SubCat2l != 1) and (data_tree.SubCat2l != 2):
                    continue
            if '_em' in analysis_bin:
                if (data_tree.SubCat2l != 3) and (data_tree.SubCat2l != 4) and (data_tree.SubCat2l != 5) and (data_tree.SubCat2l != 6):
                    continue
            if '_mm' in analysis_bin:
                if (data_tree.SubCat2l != 7) and (data_tree.SubCat2l != 8) and (data_tree.SubCat2l != 9) and (data_tree.SubCat2l != 10):
                    continue

            if '_em' in analysis_bin or '_mm' in analysis_bin:
                if 'ttHCat' in analysis_bin and data_tree.DNN_ttHnode != -999:
                    nominal_histo_.Fill(data_tree.DNN_ttHnode,data_tree.EventWeight)
                if 'ttVCat' in analysis_bin and data_tree.DNN_ttVnode != -999:
                    nominal_histo_.Fill(data_tree.DNN_ttVnode,data_tree.EventWeight)
                if 'ttJCat' in analysis_bin and data_tree.DNN_ttJnode != -999:
                    nominal_histo_.Fill(data_tree.DNN_ttJnode,data_tree.EventWeight)
            if '_ee' in analysis_bin:
                if data_tree.DNN_ttHnode != -999:
                    nominal_histo_.Fill(data_tree.DNN_ttHnode,data_tree.EventWeight)
                if data_tree.DNN_ttVnode != -999:
                    nominal_histo_.Fill(data_tree.DNN_ttVnode,data_tree.EventWeight)
                if data_tree.DNN_ttJnode != -999:
                    nominal_histo_.Fill(data_tree.DNN_ttJnode,data_tree.EventWeight)

        # When you create a file it always becomes the current directory.
        # Input file is always the last file open. To write to the output file,
        # we need to ensure output file is the current directory.
        output_file.cd()
        nominal_histo_.Write()
        nominal_histo_.SetDirectory(0)

        # Create metadata for nominal histogram. Needed to calculate rates for datacards.
        output_metadata_name = "%s/metadata_%s.txt" % (output_full_path,analysis_bin)
        with open(output_metadata_name,"a+") as output_metadata_file:
            bin_process_info = 'Analysis bin: %s , Process: %s , Nominal histogram integral: %f \n' % (analysis_bin, process_name, nominal_histo_.Integral())
            output_metadata_file.write(bin_process_info)

        # Create templates for stat uncertainty. Requires nominal histograms.
        fill_stat_unc_hists(output_file, nominal_histo_, process_name, analysis_bin)

        # Create and fill closure histograms for fakes samples
        if 'fake' in process_name:
            closure_e_up_histo_name = '%s_%s' % (process_name,'e_closUp')
            closure_e_histo_up = nominal_histo_.Clone()
            closure_e_histo_up.SetName(closure_e_up_histo_name)
            closure_e_histo_up.SetTitle(closure_e_up_histo_name)
            closure_e_histo_up.Sumw2()
            closure_e_histo_up.SetDirectory(output_file)

            closure_e_down_histo_name = '%s_%s' % (process_name,'e_closDown')
            closure_e_histo_down = nominal_histo_.Clone()
            closure_e_histo_down.SetName(closure_e_down_histo_name)
            closure_e_histo_down.SetTitle(closure_e_down_histo_name)
            closure_e_histo_down.Sumw2()
            closure_e_histo_down.SetDirectory(output_file)

            closure_m_up_histo_name = '%s_%s' % (process_name,'m_closUp')
            closure_m_histo_up = nominal_histo_.Clone()
            closure_m_histo_up.SetName(closure_m_up_histo_name)
            closure_m_histo_up.SetTitle(closure_m_up_histo_name)
            closure_m_histo_up.Sumw2()
            closure_m_histo_up.SetDirectory(output_file)

            closure_m_down_histo_name = '%s_%s' % (process_name,'m_closDown')
            closure_m_histo_down = nominal_histo_.Clone()
            closure_m_histo_down.SetName(closure_m_down_histo_name)
            closure_m_histo_down.SetTitle(closure_m_down_histo_name)
            closure_m_histo_down.Sumw2()
            closure_m_histo_down.SetDirectory(output_file)

            nbins = 10
            xmin = 0
            xmax = 1

            histo_closure_m_TT_name = 'DNNOutput_%s_closure_m_TT' % (process_name)
            histo_closure_m_QCD_name = 'DNNOutput_%s_closure_m_QCD' % (process_name)
            histo_closure_e_TT_name = 'DNNOutput_%s_closure_e_TT' % (process_name)
            histo_closure_e_QCD_name = 'DNNOutput_%s_closure_e_QCD' % (process_name)
            histo_closure_m_TT = ROOT.TH1D(histo_closure_m_TT_name,histo_closure_m_TT_name,nbins,xmin,xmax)
            histo_closure_m_QCD = ROOT.TH1D(histo_closure_m_QCD_name,histo_closure_m_QCD_name,nbins,xmin,xmax)
            histo_closure_e_TT = ROOT.TH1D(histo_closure_e_TT_name,histo_closure_e_TT_name,nbins,xmin,xmax)
            histo_closure_e_QCD = ROOT.TH1D(histo_closure_e_QCD_name,histo_closure_e_QCD_name,nbins,xmin,xmax)
            histo_closure_m_TT.Sumw2()
            histo_closure_m_QCD.Sumw2()
            histo_closure_e_TT.Sumw2()
            histo_closure_e_QCD.Sumw2()
            fill_closure_histos(process_name,analysis_bin,histo_closure_m_TT,histo_closure_m_QCD,histo_closure_e_TT,histo_closure_e_QCD)

            # Add/subtract the difference between the QCD and TT fakes templates, from the nominal templates
            # for the closure uncertainty.
            closure_e_histo_up.Add(histo_closure_e_QCD)
            closure_e_histo_up.Add(histo_closure_e_TT,-1)
            closure_e_histo_down.Add(histo_closure_e_QCD,-1)
            closure_e_histo_down.Add(histo_closure_e_TT)
            closure_m_histo_up.Add(histo_closure_m_QCD)
            closure_m_histo_up.Add(histo_closure_m_TT,-1)
            closure_m_histo_down.Add(histo_closure_m_QCD,-1)
            closure_m_histo_down.Add(histo_closure_m_TT)

            histo_closure_e_QCD.SetDirectory(0)
            histo_closure_e_TT.SetDirectory(0)
            histo_closure_m_QCD.SetDirectory(0)
            histo_closure_m_TT.SetDirectory(0)

            output_file.cd()
            closure_e_histo_up.Write()
            closure_e_histo_down.Write()
            closure_m_histo_up.Write()
            closure_m_histo_down.Write()
            closure_e_histo_up.SetDirectory(0)
            closure_m_histo_up.SetDirectory(0)
            closure_e_histo_down.SetDirectory(0)
            closure_m_histo_down.SetDirectory(0)

        # If syst list is made to be just one empty entry, the code will run for just the nominal
        if len(syst_list) == 1:
            continue

        # If syst list is filled with the various systematics then the code will create
        # templates for the systematic variations as well.
        for syst in syst_list:
            for variant in variation:
                systematic_name = syst+variant
                print 'systematic_name: ' , systematic_name

                # Histogram names must match the naming convention in the datacards
                # Namely, the $PROCESS_$SYSTEMATIC
                histo_name = '%s_%s' %(process_name,systematic_name)
                histo_ = ROOT.TH1D(histo_name,histo_name, 10, 0, 1)
                histo_.Sumw2()

                # If evaluating the JES uncertainty, atm we have separate file.
                if 'JESUp' in systematic_name:
                    # Need to use JES input files.
                    for key,value in JES_up_files_list:
                        if key == process_name:
                            print 'JES up file: %s, File: %s' % (key, value)
                            JES_input_file = value
                            data_file = TFile.Open(JES_input_file)
                            data_tree = data_file.Get("output_tree")
                        else:
                            # If not in JES files list we don;t replace the nominal.
                            # In such cases the JES is not applied (i.e. data or data driven).
                            continue
                elif 'JESDown' in systematic_name:
                    # Need to use JES input files.
                    for key,value in JES_down_files_list:
                        if key == process_name:
                            print 'JES down file: %s, File: %s' % (key, value)
                            JES_input_file = value
                            data_file = TFile.Open(JES_input_file)
                            data_tree = data_file.Get("output_tree")
                        else:
                            # If not in JES files list we don;t replace the nominal.
                            # In such cases the JES is not applied (i.e. data or data driven).
                            continue
                else:
                    for key,value in input_files_list:
                        if key == process_name:
                            print 'Key: %s, File: %s' % (key, value)
                            data_file = TFile.Open(value)
                            data_tree = data_file.Get("output_tree")
                        else:
                            continue

                print 'data tree name: ', data_file.GetName()

                for i in range(data_tree.GetEntries()):
                    data_tree.GetEntry(i)

                    if '_ee' in analysis_bin:
                        if (data_tree.SubCat2l != 1) and (data_tree.SubCat2l != 2):
                            continue
                    if '_em' in analysis_bin:
                        if (data_tree.SubCat2l != 3) and (data_tree.SubCat2l != 4) and (data_tree.SubCat2l != 5) and (data_tree.SubCat2l != 6):
                            continue
                    if '_mm' in analysis_bin:
                        if (data_tree.SubCat2l != 7) and (data_tree.SubCat2l != 8) and (data_tree.SubCat2l != 9) and (data_tree.SubCat2l != 10):
                            continue

                    event_weight = 1.0

                    if 'JESDown' in systematic_name:
                        event_weight = data_tree.EventWeight
                    if 'JESUp' in systematic_name:
                        event_weight = data_tree.EventWeight

                    pileupweight = 1.0
                    if data_tree.puWeight != 0:
                        pileupweight = data_tree.puWeight

                    if 'PUUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.puWeight_SysUp/pileupweight
                    if 'PUDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.puWeight_SysDown/pileupweight

                    triggersf = 1.0
                    if data_tree.TriggerSF != 0:
                        triggersf = data_tree.TriggerSF

                    if 'TrigUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.TriggerSF_SysUp/triggersf
                    if 'TrigDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.TriggerSF_SysDown/triggersf

                    genweight = 1.0
                    if data_tree.EVENT_genWeight != 0:
                        genweight = data_tree.EVENT_genWeight

                    if 'genWeight_muFDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.genWeight_muF0p5/genweight
                    if 'genWeight_muFUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.genWeight_muF2/genweight

                    if 'genWeight_muRDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.genWeight_muR0p5/genweight
                    if 'genWeight_muRUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.genWeight_muR2/genweight

                    eleloosesf = 1.0
                    if data_tree.elelooseSF != 0:
                        eleloosesf = data_tree.elelooseSF

                    if 'elelooseSFDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.elelooseSF_SysDown/eleloosesf
                    if 'elelooseSFUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.elelooseSF_SysUp/eleloosesf

                    eletightsf = 1.0
                    if data_tree.eletightSF != 0:
                        eletightsf = data_tree.eletightSF

                    if 'eletightSFDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.eletightSF_SysDown/eletightsf
                    if 'eletightSFUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.eletightSF_SysUp/eletightsf

                    muloosesf = 1.0
                    if data_tree.mulooseSF != 0:
                        muloosesf = data_tree.mulooseSF

                    if 'mulooseSFDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.mulooseSF_SysDown/muloosesf
                    if 'mulooseSFUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.mulooseSF_SysUp/muloosesf

                    mutightsf = 1.0
                    if data_tree.mutightSF != 0:
                        mutightsf = data_tree.mutightSF

                    if 'mutightSFDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.mutightSF_SysDown/mutightsf
                    if 'mutightSFUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.mutightSF_SysUp/mutightsf

                    # Currently not dividing bweight central value if = 0
                    # Simply apply variation. Not sure if this is neceassarily
                    # the correct treatment.
                    bweight = 1.0
                    if data_tree.bWeight_central != 0:
                        bweight = data_tree.bWeight_central

                    if 'bWeight_hfstats1Up' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_up_hfstats1/bweight
                    if 'bWeight_hfstats1Down' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_down_hfstats1/bweight

                    if 'bWeight_hfstats2Up' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_up_hfstats2/bweight
                    if 'bWeight_hfstats2Down' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_down_hfstats2/bweight

                    if 'bWeight_cferr1Up' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_up_cferr1/bweight
                    if 'bWeight_cferr1Down' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_down_cferr1/bweight

                    if 'bWeight_cferr2Up' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_up_cferr2/bweight
                    if 'bWeight_cferr2Down' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_down_cferr2/bweight

                    if 'bWeight_jesUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_up_jes/bweight
                    if 'bWeight_jesDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_down_jes/bweight

                    if 'bWeight_lfUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_up_lf/bweight
                    if 'bWeight_lfDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_down_lf/bweight

                    if 'bWeight_hfUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_up_hf/bweight
                    if 'bWeight_hfDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.bWeight_down_hf/bweight

                    # Only used if sample = Flips
                    chargemisid = 1.0
                    if data_tree.ChargeMis != 0:
                        chargemisid = data_tree.ChargeMis

                    if 'ChargeMisDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.ChargeMis_SysDown/chargemisid
                    if 'ChargeMisUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.ChargeMis_SysUp/chargemisid

                    fakerate = 1.0
                    if data_tree.FakeRate != 0:
                        fakerate = data_tree.FakeRate

                    if 'FakeRate_m_normUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_m_up/fakerate
                    if 'FakeRate_m_normDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_m_down/fakerate
                    if 'FakeRate_m_ptUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_m_pt1/fakerate
                    if 'FakeRate_m_ptDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_m_pt2/fakerate
                    if 'FakeRate_m_beUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_m_be1/fakerate
                    if 'FakeRate_m_beDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_m_be2/fakerate
                    if 'FakeRate_e_normUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_e_up/fakerate
                    if 'FakeRate_e_normDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_e_down/fakerate
                    if 'FakeRate_e_ptUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_e_pt1/fakerate
                    if 'FakeRate_e_ptDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_e_pt2/fakerate
                    if 'FakeRate_e_beUp' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_e_be1/fakerate
                    if 'FakeRate_e_beDown' in systematic_name:
                        event_weight = data_tree.EventWeight*data_tree.FakeRate_e_be2/fakerate

                    if '_em' in analysis_bin or '_mm' in analysis_bin:
                        if 'ttHCat' in analysis_bin and data_tree.DNN_ttHnode != -999:
                            histo_.Fill(data_tree.DNN_ttHnode,event_weight)
                        if 'ttVCat' in analysis_bin and data_tree.DNN_ttVnode != -999:
                            histo_.Fill(data_tree.DNN_ttVnode,event_weight)
                        if 'ttJCat' in analysis_bin and data_tree.DNN_ttJnode != -999:
                            histo_.Fill(data_tree.DNN_ttJnode,event_weight)
                    if '_ee' in analysis_bin:
                        if data_tree.DNN_ttHnode != -999:
                            histo_.Fill(data_tree.DNN_ttHnode,event_weight)
                        if data_tree.DNN_ttVnode != -999:
                            histo_.Fill(data_tree.DNN_ttVnode,event_weight)
                        if data_tree.DNN_ttJnode != -999:
                            histo_.Fill(data_tree.DNN_ttJnode,event_weight)
                output_file.cd()
                histo_.Write()
                histo_.SetDirectory(0)
    output_file.Close()
    print 'Job complete.'
    sys.exit(0)

main()
