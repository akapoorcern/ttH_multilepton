#################################
#     DNN_ResponsePlotter.py
#     Joshuha Thomas-Wilsker
#       IHEP Bejing, CERN
#################################
# Plotting script using output
# root files from TMVA training.
#################################

import ROOT
from ROOT import TFile, TTree, gDirectory, gROOT, TH1, TF1, TProfile, TProfile2D, TLegend
import os
import optparse

#import numpy as np
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt

#def plot_ROCS(signal_hist, bckg0_hist, bckg1_hist):

def plot_DNNResponse(TrainTree, TestTree, classifier_suffix):
    # Makes plot from TestTree/TrainTree distributions. Plots made from
    # combined response values from each node for a single sample.
    classifier_parent_dir = 'MultiClass_DNN_%s' % (classifier_suffix)
    # Declare and define new hitogram objects
    Histo_training_ttH_DNNResponse = ROOT.TH1D('Histo_training_ttH_DNNResponse','ttH sample (Train)',40,0.0,1.0)
    Histo_training_ttJets_DNNResponse = ROOT.TH1D('Histo_training_ttJets_DNNResponse','ttJets sample (Train)',40,0.0,1.0)
    Histo_training_ttV_DNNResponse = ROOT.TH1D('Histo_training_ttV_DNNResponse','ttV sample (Train)',40,0.0,1.0)

    Histo_testing_ttH_DNNRepsonse = ROOT.TH1D('Histo_testing_ttH_DNNRepsonse','ttH sample (Test)',40,0.0,1.0)
    Histo_testing_ttJets_DNNResponse = ROOT.TH1D('Histo_testing_ttJets_DNNResponse','ttJets sample (Test)',40,0.0,1.0)
    Histo_testing_ttV_DNNResponse = ROOT.TH1D('Histo_testing_ttV_DNNResponse','ttV sample (Test)',40,0.0,1.0)

    # Now lets project the tree information into those histograms
    # My current understanding is that the three branches beneath
    # represent the response (i.e.)
    TrainTree.Project("Histo_training_ttH_DNNResponse","DNN.ttH")
    TrainTree.Project("Histo_training_ttV_DNNResponse","DNN.ttV")
    TrainTree.Project("Histo_training_ttJets_DNNResponse","DNN.ttJets")

    TestTree.Project("Histo_testing_ttH_DNNRepsonse","DNN.ttH")
    TestTree.Project("Histo_testing_ttV_DNNResponse","DNN.ttV")
    TestTree.Project("Histo_testing_ttJets_DNNResponse","DNN.ttJets")

    Histo_training_ttH_DNNResponse.Sumw2()
    Histo_training_ttJets_DNNResponse.Sumw2()
    Histo_training_ttV_DNNResponse.Sumw2()
    Histo_testing_ttH_DNNRepsonse.Sumw2()
    Histo_testing_ttJets_DNNResponse.Sumw2()
    Histo_testing_ttV_DNNResponse.Sumw2()

    c1 = ROOT.TCanvas("c1","",1000,700)
    p1 = ROOT.TPad("p1","p1",0.0,0.2,1.0,1.0)
    p1.Draw()
    p1.SetRightMargin(0.1)
    p1.SetLeftMargin(0.1)
    p1.SetTopMargin(0.1)
    p1.SetBottomMargin(0.05)
    p1.SetGridx(True)
    p1.SetGridy(True)
    p1.cd()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    # Set titles
    Histo_training_ttJets_DNNResponse.GetYaxis().SetTitle("Counts/Bin")

    # Set ttH style
    Histo_training_ttH_DNNResponse.SetLineColor(2)
    Histo_training_ttH_DNNResponse.SetMarkerColor(2)
    Histo_training_ttH_DNNResponse.SetFillColor(2)
    Histo_training_ttH_DNNResponse.SetFillStyle(3001)

    Histo_testing_ttH_DNNRepsonse.SetLineColor(2)
    Histo_testing_ttH_DNNRepsonse.SetMarkerColor(2)
    Histo_testing_ttH_DNNRepsonse.SetFillColor(2)
    Histo_testing_ttH_DNNRepsonse.SetMarkerStyle(20)


    # Set ttJets style
    Histo_training_ttJets_DNNResponse.SetLineColor(4)
    Histo_training_ttJets_DNNResponse.SetMarkerColor(4)
    Histo_training_ttJets_DNNResponse.SetFillColor(4)
    Histo_training_ttJets_DNNResponse.SetFillStyle(3001)

    Histo_testing_ttJets_DNNResponse.SetLineColor(4)
    Histo_testing_ttJets_DNNResponse.SetMarkerColor(4)
    Histo_testing_ttJets_DNNResponse.SetFillColor(4)
    Histo_testing_ttJets_DNNResponse.SetMarkerStyle(20)

    # Set ttV style
    Histo_training_ttV_DNNResponse.SetLineColor(3)
    Histo_training_ttV_DNNResponse.SetMarkerColor(3)
    Histo_training_ttV_DNNResponse.SetFillColor(3)
    Histo_training_ttV_DNNResponse.SetFillStyle(3001)

    Histo_testing_ttV_DNNResponse.SetLineColor(3)
    Histo_testing_ttV_DNNResponse.SetMarkerColor(3)
    Histo_testing_ttV_DNNResponse.SetFillColor(3)
    Histo_testing_ttV_DNNResponse.SetMarkerStyle(20)
    Histo_training_ttJets_DNNResponse.SetAxisRange(0.,10000.,"Y")
    # Draw the objects
    Histo_training_ttJets_DNNResponse.Draw("HIST")
    Histo_training_ttH_DNNResponse.Draw("HISTSAME")
    Histo_training_ttV_DNNResponse.Draw("HISTSAME")
    Histo_testing_ttH_DNNRepsonse.Draw("EPSAME")
    Histo_testing_ttJets_DNNResponse.Draw("EPSAME")
    Histo_testing_ttV_DNNResponse.Draw("EPSAME")

    legend = TLegend(0.8,  0.7,  0.99,  0.99)
    legend.AddEntry(Histo_training_ttH_DNNResponse,"ttH events (train)")
    legend.AddEntry(Histo_training_ttV_DNNResponse,"ttV events (train)")
    legend.AddEntry(Histo_training_ttJets_DNNResponse,"tt+jets events (train)")
    legend.AddEntry(Histo_testing_ttH_DNNRepsonse,"ttH events (test)")
    legend.AddEntry(Histo_testing_ttV_DNNResponse,"ttV events (test)")
    legend.AddEntry(Histo_testing_ttJets_DNNResponse,"tt+jets events (test)")
    legend.Draw("same")

    # Add custom title
    l1=ROOT.TLatex()
    l1.SetNDC();
    l1.DrawLatex(0.36,0.94,"Multiclass DNN Response")

    c1.cd()
    p2 = ROOT.TPad("p2","p2",0.0,0.0,1.0,0.2)
    p2.Draw()
    p2.SetLeftMargin(0.1)
    p2.SetRightMargin(0.1)
    p2.SetTopMargin(0.05)
    p2.SetBottomMargin(0.4)
    p2.SetGridx(True)
    p2.SetGridy(True)
    p2.cd()

    ratioframe_ttH = Histo_training_ttH_DNNResponse.Clone('ratioframe_ttH')
    ratioframe_ttH.Divide(Histo_testing_ttH_DNNRepsonse)
    ratioframe_ttH.GetYaxis().SetTitle('Train/Test')
    ratioframe_ttH.GetYaxis().SetRangeUser(0.46,1.54)
    ratioframe_ttH.GetYaxis().SetNdivisions(6)
    ratioframe_ttH.GetYaxis().SetLabelSize(0.12)
    ratioframe_ttH.GetYaxis().SetTitleSize(0.12)
    ratioframe_ttH.GetYaxis().SetTitleOffset(0.2)
    ratioframe_ttH.GetXaxis().SetTitle('DNN Response')
    ratioframe_ttH.GetXaxis().SetLabelSize(0.15)
    ratioframe_ttH.GetXaxis().SetTitleSize(0.15)
    ratioframe_ttH.GetXaxis().SetTitleOffset(1.)
    ratioframe_ttH.SetFillStyle(0)
    ratioframe_ttH.SetMarkerStyle(2)
    ratioframe_ttH.SetMarkerColor(2)
    ratioframe_ttH.Draw('P')

    ratioframe_ttV = Histo_training_ttV_DNNResponse.Clone('ratioframe_ttV')
    ratioframe_ttV.Divide(Histo_testing_ttV_DNNResponse)
    ratioframe_ttV.SetMarkerStyle(2)
    ratioframe_ttV.SetMarkerColor(3)
    ratioframe_ttV.Draw("sameP")

    ratioframe_ttJets = Histo_training_ttJets_DNNResponse.Clone('ratioframe_ttJets')
    ratioframe_ttJets.Divide(Histo_testing_ttJets_DNNResponse)
    ratioframe_ttJets.SetMarkerStyle(2)
    ratioframe_ttJets.SetMarkerColor(4)
    ratioframe_ttJets.Draw("sameP")

    c1.cd()
    c1.Modified()
    c1.Update()

    # Finally, draw the figure
    outfile_name = '%s/plots/MCDNN_overtraintest_%s.pdf' % (classifier_parent_dir,classifier_suffix)
    c1.Print(outfile_name,'pdf')

def plot_node_response(input_root, node, classifier_suffix):
    classifier_parent_dir = 'MultiClass_DNN_%s' % (classifier_suffix)
    if node == 'tth':
        histo_DNN_response_ttHsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttH')
        histo_DNN_response_ttVsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttV')
        histo_DNN_response_ttJetssample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttJets')
        histo_DNN_response_ttHsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttH')
        histo_DNN_response_ttVsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttV')
        histo_DNN_response_ttJetssample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttJets')
        '''
        histo_DNN_response_ttHsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttH')
        histo_DNN_response_ttVsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttH')
        histo_DNN_response_ttJetssample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttH')
        histo_DNN_response_ttHsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttH')
        histo_DNN_response_ttVsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttH')
        histo_DNN_response_ttJetssample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttH')
        '''
    elif node == 'ttV':
        histo_DNN_response_ttHsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttH')
        histo_DNN_response_ttVsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttV')
        histo_DNN_response_ttJetssample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttJets')
        histo_DNN_response_ttHsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttH')
        histo_DNN_response_ttVsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttV')
        histo_DNN_response_ttJetssample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttJets')
        '''
        histo_DNN_response_ttHsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttV')
        histo_DNN_response_ttVsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttV')
        histo_DNN_response_ttJetssample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttV')
        histo_DNN_response_ttHsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttV')
        histo_DNN_response_ttVsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttV')
        histo_DNN_response_ttJetssample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttV')
        '''
    elif node == 'ttJets':
        histo_DNN_response_ttHsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttH')
        histo_DNN_response_ttVsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttV')
        histo_DNN_response_ttJetssample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttJets')
        histo_DNN_response_ttHsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttH')
        histo_DNN_response_ttVsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttV')
        histo_DNN_response_ttJetssample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttJets')
        '''
        histo_DNN_response_ttHsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttJets')
        histo_DNN_response_ttVsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttJets')
        histo_DNN_response_ttJetssample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttJets')
        histo_DNN_response_ttHsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttJets')
        histo_DNN_response_ttVsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttJets')
        histo_DNN_response_ttJetssample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttJets')
        '''

    histo_DNN_response_ttHsample_test.Sumw2()
    histo_DNN_response_ttVsample_test.Sumw2()
    histo_DNN_response_ttJetssample_test.Sumw2()

    histo_DNN_response_ttHsample_train.Sumw2()
    histo_DNN_response_ttVsample_train.Sumw2()
    histo_DNN_response_ttJetssample_train.Sumw2()

    c1 = ROOT.TCanvas("c1","c1",900,700)
    p1 = ROOT.TPad("p1","p1", 0.0,0.2,1.0,1.0)
    p1.Draw()
    p1.SetBottomMargin(0.1)
    p1.SetTopMargin(0.1)
    p1.SetLeftMargin(0.1)
    p1.SetRightMargin(0.1)
    p1.SetGridx(True)
    p1.SetGridy(True)
    p1.cd()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    histo_DNN_response_ttHsample_test.SetLineColor(2)
    histo_DNN_response_ttHsample_test.SetMarkerColor(2)
    histo_DNN_response_ttHsample_test.SetMarkerStyle(20)

    histo_DNN_response_ttHsample_train.SetLineColor(2)
    histo_DNN_response_ttHsample_train.SetMarkerColor(2)
    histo_DNN_response_ttHsample_train.SetFillColor(2)
    histo_DNN_response_ttHsample_train.SetFillStyle(3001)

    histo_DNN_response_ttVsample_test.SetLineColor(3)
    histo_DNN_response_ttVsample_test.SetMarkerColor(3)
    histo_DNN_response_ttVsample_test.SetMarkerStyle(20)

    histo_DNN_response_ttVsample_train.SetLineColor(3)
    histo_DNN_response_ttVsample_train.SetMarkerColor(3)
    histo_DNN_response_ttVsample_train.SetFillColor(3)
    histo_DNN_response_ttVsample_train.SetFillStyle(3001)

    histo_DNN_response_ttJetssample_test.SetLineColor(4)
    histo_DNN_response_ttJetssample_test.SetMarkerColor(4)
    histo_DNN_response_ttJetssample_test.SetMarkerStyle(20)

    histo_DNN_response_ttJetssample_train.SetLineColor(4)
    histo_DNN_response_ttJetssample_train.SetMarkerColor(4)
    histo_DNN_response_ttJetssample_train.SetFillColor(4)
    histo_DNN_response_ttJetssample_train.SetFillStyle(3001)

    histo_DNN_response_ttHsample_test.GetYaxis().SetTitle("Counts/Bin (normalised)")

    legend = TLegend(0.8,  0.7,  0.99,  0.99)
    legend.AddEntry(histo_DNN_response_ttHsample_train,"ttH events (train)")
    legend.AddEntry(histo_DNN_response_ttVsample_train,"ttV events (train)")
    legend.AddEntry(histo_DNN_response_ttJetssample_train,"tt+jets events (train)")
    legend.AddEntry(histo_DNN_response_ttHsample_test,"ttH events (test)")
    legend.AddEntry(histo_DNN_response_ttVsample_test,"ttV events (test)")
    legend.AddEntry(histo_DNN_response_ttJetssample_test,"tt+jets events (test)")

    histo_DNN_response_ttHsample_test.SetAxisRange(0.,4.,"Y")
    histo_DNN_response_ttHsample_test.Draw('EP')
    histo_DNN_response_ttVsample_test.Draw('EPsame')
    histo_DNN_response_ttJetssample_test.Draw('EPsame')
    histo_DNN_response_ttHsample_train.Draw('HISTsame')
    histo_DNN_response_ttVsample_train.Draw('HISTsame')
    histo_DNN_response_ttJetssample_train.Draw('HISTsame')
    legend.Draw("sameP")

    c1.cd()
    p2 = ROOT.TPad("p2","p2",0.0,0.0,1.0,0.2)
    p2.Draw()
    p2.SetLeftMargin(0.1)
    p2.SetRightMargin(0.1)
    p2.SetTopMargin(0.05)
    p2.SetBottomMargin(0.4)
    p2.SetGridx(True)
    p2.SetGridy(True)
    p2.cd()

    ratioframe_ttH = histo_DNN_response_ttHsample_train.Clone('ratioframe_ttH')
    ratioframe_ttH.Divide(histo_DNN_response_ttHsample_test)
    ratioframe_ttH.GetYaxis().SetTitle('Train/Test')
    ratioframe_ttH.GetYaxis().SetRangeUser(0.46,1.54)
    ratioframe_ttH.GetYaxis().SetNdivisions(6)
    ratioframe_ttH.GetYaxis().SetLabelSize(0.12)
    ratioframe_ttH.GetYaxis().SetTitleSize(0.12)
    ratioframe_ttH.GetYaxis().SetTitleOffset(0.2)
    ratioframe_ttH.GetXaxis().SetTitle('DNN Response')
    ratioframe_ttH.GetXaxis().SetLabelSize(0.15)
    ratioframe_ttH.GetXaxis().SetTitleSize(0.15)
    ratioframe_ttH.GetXaxis().SetTitleOffset(1.)
    ratioframe_ttH.SetFillStyle(0)
    ratioframe_ttH.SetMarkerStyle(2)
    ratioframe_ttH.SetMarkerColor(2)
    ratioframe_ttH.Draw('P')

    ratioframe_ttV = histo_DNN_response_ttVsample_train.Clone('ratioframe_ttV')
    ratioframe_ttV.Divide(histo_DNN_response_ttVsample_test)
    ratioframe_ttV.SetMarkerStyle(2)
    ratioframe_ttV.SetMarkerColor(3)
    ratioframe_ttV.Draw("sameP")

    ratioframe_ttJets = histo_DNN_response_ttJetssample_train.Clone('ratioframe_ttJets')
    ratioframe_ttJets.Divide(histo_DNN_response_ttJetssample_test)
    ratioframe_ttJets.SetMarkerStyle(2)
    ratioframe_ttJets.SetMarkerColor(4)
    ratioframe_ttJets.Draw("sameP")

    c1.cd()
    c1.Modified()
    c1.Update()

    outfile_name = '%s/plots/MCDNN_Response_%s-%s.pdf' % (classifier_parent_dir, classifier_suffix , node)
    c1.Print(outfile_name,'pdf')
    c1.Clear()

def main():
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify inputs from network training',      default='2HLs_relu',        type='string')
    (opt, args) = parser.parse_args()

    classifier_suffix = opt.input_suffix

    classifier_parent_dir = 'MultiClass_DNN_%s' % (classifier_suffix)
    classifier_plots_dir = classifier_parent_dir+"/plots"
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    input_name = 'ttHML_MCDNN_%s.root' % (classifier_suffix)
    input_root = TFile.Open(input_name)
    traintree_pathname = "MultiClass_DNN_%s/TrainTree" % (classifier_suffix)
    testtree_pathname = "MultiClass_DNN_%s/TestTree" % (classifier_suffix)
    # Fetch the trees of events from the root file
    TrainTree = input_root.Get(traintree_pathname)
    TestTree = input_root.Get(testtree_pathname)
    plot_DNNResponse(TrainTree,TestTree, classifier_suffix)
    plot_node_response(input_root,'tth',classifier_suffix)
    plot_node_response(input_root,'ttV',classifier_suffix)
    plot_node_response(input_root,'ttJets',classifier_suffix)

main()
