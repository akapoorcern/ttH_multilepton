#################################
#     DNN_ResponsePlotter.py
#     Joshuha Thomas-Wilsker
#       IHEP Bejing, CERN
#################################
# Plotting script using output
# root files from TMVA training.
#################################

import ROOT
from ROOT import TFile, TTree, gDirectory, gROOT, TH1, TF1, TProfile, TProfile2D, TLegend, TLatex
import os
import optparse
from scipy import stats
import numpy as np

def GetSeparation(hist_sig, hist_bckg):

    # compute "separation" defined as
    # <s2> = (1/2) Int_-oo..+oo { (S(x) - B(x))^2/(S(x) + B(x)) dx }
    separation = 0;
    # sanity check: signal and background histograms must have same number of bins and same limits
    if hist_sig.GetNbinsX() != hist_bckg.GetNbinsX():
        print 'Number of bins different for sig. and bckg'

    nBins = hist_sig.GetNbinsX()
    dX = (hist_sig.GetXaxis().GetXmax() - hist_sig.GetXaxis().GetXmin()) / hist_sig.GetNbinsX()
    nS = hist_sig.GetSumOfWeights()*dX
    nB = hist_bckg.GetSumOfWeights()*dX

    if nS == 0:
        print 'WARNING: no signal weights'
    if nB == 0:
        print 'WARNING: no bckg weights'

    for i in xrange(nBins):
        sig_bin_norm = hist_sig.GetBinContent(i)/nS
        bckg_bin_norm = hist_bckg.GetBinContent(i)/nB
        # Separation:
        if(sig_bin_norm+bckg_bin_norm > 0):
            separation += 0.5 * ((sig_bin_norm - bckg_bin_norm) * (sig_bin_norm - bckg_bin_norm)) / (sig_bin_norm + bckg_bin_norm)
    separation *= dX
    return separation

def plot_DNNResponse(TrainTree, TestTree, classifier_suffix):
    # Makes plot from TestTree/TrainTree distributions. Plots made from
    # combined response values from each node for a single sample.
    classifier_parent_dir = 'MultiClass_DNN_allJets_%s' % (classifier_suffix)
    # Declare and define new hitogram objects
    Histo_training_ttH_DNNResponse = ROOT.TH1D('Histo_training_ttH_DNNResponse','ttH sample (Train)',40,0.0,1.0)
    Histo_training_ttJets_DNNResponse = ROOT.TH1D('Histo_training_ttJets_DNNResponse','ttJets sample (Train)',40,0.0,1.0)
    Histo_training_ttV_DNNResponse = ROOT.TH1D('Histo_training_ttV_DNNResponse','ttV sample (Train)',40,0.0,1.0)

    Histo_testing_ttH_DNNResponse = ROOT.TH1D('Histo_testing_ttH_DNNResponse','ttH sample (Test)',40,0.0,1.0)
    Histo_testing_ttJets_DNNResponse = ROOT.TH1D('Histo_testing_ttJets_DNNResponse','ttJets sample (Test)',40,0.0,1.0)
    Histo_testing_ttV_DNNResponse = ROOT.TH1D('Histo_testing_ttV_DNNResponse','ttV sample (Test)',40,0.0,1.0)

    # Now lets project the tree information into those histograms
    # My current understanding is that the three branches beneath
    # represent the response (i.e.)
    TrainTree.Project("Histo_training_ttH_DNNResponse","DNN.ttH")
    TrainTree.Project("Histo_training_ttV_DNNResponse","DNN.ttV")
    TrainTree.Project("Histo_training_ttJets_DNNResponse","DNN.ttJets")

    TestTree.Project("Histo_testing_ttH_DNNResponse","DNN.ttH")
    TestTree.Project("Histo_testing_ttV_DNNResponse","DNN.ttV")
    TestTree.Project("Histo_testing_ttJets_DNNResponse","DNN.ttJets")

    Histo_training_ttH_DNNResponse.Sumw2()
    Histo_training_ttJets_DNNResponse.Sumw2()
    Histo_training_ttV_DNNResponse.Sumw2()
    Histo_testing_ttH_DNNResponse.Sumw2()
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
    Histo_training_ttH_DNNResponse.GetYaxis().SetTitle("(1/N)dN/dX")
    Histo_training_ttJets_DNNResponse.GetYaxis().SetTitle("(1/N)dN/dX")
    Histo_training_ttV_DNNResponse.GetYaxis().SetTitle("(1/N)dN/dX")
    Histo_testing_ttH_DNNResponse.GetYaxis().SetTitle("(1/N)dN/dX")
    Histo_testing_ttJets_DNNResponse.GetYaxis().SetTitle("(1/N)dN/dX")
    Histo_testing_ttV_DNNResponse.GetYaxis().SetTitle("(1/N)dN/dX")

    # Set ttH style
    Histo_training_ttH_DNNResponse.SetLineColor(2)
    Histo_training_ttH_DNNResponse.SetMarkerColor(2)
    Histo_training_ttH_DNNResponse.SetFillColor(2)
    Histo_training_ttH_DNNResponse.SetFillStyle(3001)

    Histo_testing_ttH_DNNResponse.SetLineColor(2)
    Histo_testing_ttH_DNNResponse.SetMarkerColor(2)
    Histo_testing_ttH_DNNResponse.SetMarkerStyle(20)

    # Set ttJets style
    Histo_training_ttJets_DNNResponse.SetLineColor(4)
    Histo_training_ttJets_DNNResponse.SetMarkerColor(4)
    Histo_training_ttJets_DNNResponse.SetFillColor(4)
    Histo_training_ttJets_DNNResponse.SetFillStyle(3001)

    Histo_testing_ttJets_DNNResponse.SetLineColor(4)
    Histo_testing_ttJets_DNNResponse.SetMarkerColor(4)
    Histo_testing_ttJets_DNNResponse.SetMarkerStyle(20)

    # Set ttV style
    Histo_training_ttV_DNNResponse.SetLineColor(3)
    Histo_training_ttV_DNNResponse.SetMarkerColor(3)
    Histo_training_ttV_DNNResponse.SetFillColor(3)
    Histo_training_ttV_DNNResponse.SetFillStyle(3001)

    Histo_testing_ttV_DNNResponse.SetLineColor(3)
    Histo_testing_ttV_DNNResponse.SetMarkerColor(3)
    Histo_testing_ttV_DNNResponse.SetMarkerStyle(20)

    dx_ttH_train = (Histo_training_ttH_DNNResponse.GetXaxis().GetXmax() - Histo_training_ttH_DNNResponse.GetXaxis().GetXmin()) / Histo_training_ttH_DNNResponse.GetNbinsX()
    ttH_train_norm = 1./Histo_training_ttH_DNNResponse.GetSumOfWeights()/dx_ttH_train
    Histo_training_ttH_DNNResponse.Scale(ttH_train_norm)

    dx_ttH_test = (Histo_testing_ttH_DNNResponse.GetXaxis().GetXmax() - Histo_testing_ttH_DNNResponse.GetXaxis().GetXmin()) / Histo_testing_ttH_DNNResponse.GetNbinsX()
    ttH_test_norm = 1./Histo_testing_ttH_DNNResponse.GetSumOfWeights()/dx_ttH_test
    Histo_testing_ttH_DNNResponse.Scale(ttH_test_norm)

    dx_ttV_train = (Histo_training_ttV_DNNResponse.GetXaxis().GetXmax() - Histo_training_ttV_DNNResponse.GetXaxis().GetXmin()) / Histo_training_ttV_DNNResponse.GetNbinsX()
    ttV_train_norm = 1./Histo_training_ttV_DNNResponse.GetSumOfWeights()/dx_ttV_train
    Histo_training_ttV_DNNResponse.Scale(ttV_train_norm)

    dx_ttV_test = (Histo_testing_ttV_DNNResponse.GetXaxis().GetXmax() - Histo_testing_ttV_DNNResponse.GetXaxis().GetXmin()) / Histo_testing_ttV_DNNResponse.GetNbinsX()
    ttV_test_norm = 1./Histo_testing_ttV_DNNResponse.GetSumOfWeights()/dx_ttV_test
    Histo_testing_ttV_DNNResponse.Scale(ttV_test_norm)

    dx_ttJ_train = (Histo_training_ttJets_DNNResponse.GetXaxis().GetXmax() - Histo_training_ttJets_DNNResponse.GetXaxis().GetXmin()) / Histo_training_ttJets_DNNResponse.GetNbinsX()
    ttJ_train_norm = 1./Histo_training_ttJets_DNNResponse.GetSumOfWeights()/dx_ttJ_train
    Histo_training_ttJets_DNNResponse.Scale(ttJ_train_norm)

    dx_ttJ_test = (Histo_testing_ttJets_DNNResponse.GetXaxis().GetXmax() - Histo_testing_ttJets_DNNResponse.GetXaxis().GetXmin()) / Histo_testing_ttJets_DNNResponse.GetNbinsX()
    ttJ_test_norm = 1./Histo_testing_ttJets_DNNResponse.GetSumOfWeights()/dx_ttJ_test
    Histo_testing_ttJets_DNNResponse.Scale(ttJ_test_norm)

    maxyaxis = max(Histo_training_ttH_DNNResponse.GetMaximum(), Histo_testing_ttH_DNNResponse.GetMaximum(), Histo_training_ttJets_DNNResponse.GetMaximum(), Histo_testing_ttJets_DNNResponse.GetMaximum(), Histo_training_ttV_DNNResponse.GetMaximum(), Histo_training_ttV_DNNResponse.GetMaximum())
    Histo_training_ttH_DNNResponse.SetAxisRange(0.,maxyaxis+(maxyaxis/4),"Y")

    # Draw the objects
    Histo_training_ttH_DNNResponse.Draw("HIST")
    Histo_training_ttJets_DNNResponse.Draw("HISTSAME")
    Histo_training_ttV_DNNResponse.Draw("HISTSAME")
    Histo_testing_ttH_DNNResponse.Draw("EPSAME")
    Histo_testing_ttJets_DNNResponse.Draw("EPSAME")
    Histo_testing_ttV_DNNResponse.Draw("EPSAME")

    legend = TLegend(0.8,  0.7,  0.99,  0.99)
    legend.AddEntry(Histo_training_ttH_DNNResponse,"ttH node (train)")
    legend.AddEntry(Histo_training_ttV_DNNResponse,"ttV node (train)")
    legend.AddEntry(Histo_training_ttJets_DNNResponse,"tt+jets node (train)")
    legend.AddEntry(Histo_testing_ttH_DNNResponse,"ttH node (test)")
    legend.AddEntry(Histo_testing_ttV_DNNResponse,"ttV node (test)")
    legend.AddEntry(Histo_testing_ttJets_DNNResponse,"tt+jets node (test)")
    legend.Draw("same")

    # Test statistical significance between training and testing distributions.

    ttHnode_training_response_array = []
    for n in xrange(Histo_training_ttH_DNNResponse.GetNbinsX()):
        ttHnode_training_response_array.append(Histo_training_ttH_DNNResponse.GetBinContent(n))
    ttHnode_training_response_nparray = np.array(ttHnode_training_response_array)

    ttHnode_testing_response_array = []
    for n in xrange(Histo_testing_ttH_DNNResponse.GetNbinsX()):
        ttHnode_testing_response_array.append(Histo_testing_ttH_DNNResponse.GetBinContent(n))
    ttHnode_testing_response_nparray = np.array(ttHnode_testing_response_array)

    ttVnode_training_response_array = []
    for n in xrange(Histo_training_ttV_DNNResponse.GetNbinsX()):
        ttVnode_training_response_array.append(Histo_training_ttV_DNNResponse.GetBinContent(n))
    ttVnode_training_response_nparray = np.array(ttVnode_training_response_array)

    ttVnode_testing_response_array = []
    for n in xrange(Histo_testing_ttV_DNNResponse.GetNbinsX()):
        ttVnode_testing_response_array.append(Histo_testing_ttV_DNNResponse.GetBinContent(n))
    ttVnode_testing_response_nparray = np.array(ttVnode_testing_response_array)

    ttJetsnode_training_response_array = []
    for n in xrange(Histo_training_ttJets_DNNResponse.GetNbinsX()):
        ttJetsnode_training_response_array.append(Histo_training_ttJets_DNNResponse.GetBinContent(n))
    ttJetsnode_training_response_nparray = np.array(ttJetsnode_training_response_array)

    ttJetsnode_testing_response_array = []
    for n in xrange(Histo_testing_ttJets_DNNResponse.GetNbinsX()):
        ttJetsnode_testing_response_array.append(Histo_testing_ttJets_DNNResponse.GetBinContent(n))
    ttJetsnode_testing_response_nparray = np.array(ttJetsnode_testing_response_array)

    print 'ttHnode_training_response_nparray: ', ttHnode_training_response_nparray
    print 'ttHnode_testing_response_nparray: ', ttHnode_testing_response_nparray

    ks_teststat_ttHnode,ks_pval_ttHnode = stats.ks_2samp(ttHnode_training_response_nparray, ttHnode_testing_response_nparray)
    ks_teststat_ttVnode,ks_pval_ttVnode = stats.ks_2samp(ttVnode_training_response_nparray, ttVnode_testing_response_nparray)
    ks_teststat_ttJetsnode,ks_pval_ttJetsnode = stats.ks_2samp(ttJetsnode_training_response_nparray, ttJetsnode_testing_response_nparray)

    anderson_ttHnode = stats.anderson_ksamp([ttHnode_training_response_nparray, ttHnode_testing_response_nparray])
    anderson_ttVnode = stats.anderson_ksamp([ttVnode_training_response_nparray, ttVnode_testing_response_nparray])
    anderson_ttJetsnode = stats.anderson_ksamp([ttJetsnode_training_response_nparray, ttJetsnode_testing_response_nparray])

    ks_test_statistic_ttH = "#scale[0.5]{ttH node KS test stat. = %f}" % ks_teststat_ttHnode
    ks_test_statistic_ttV = "#scale[0.5]{ttV node KS test stat. = %f}" % ks_teststat_ttVnode
    ks_test_statistic_ttJets = "#scale[0.5]{ttJets node KS test stat. = %f}" % ks_teststat_ttJetsnode
    ks_pval_ttH = "#scale[0.5]{ttH node KS p-value = %f}" % ks_pval_ttHnode
    ks_pval_ttV = "#scale[0.5]{ttV node KS p-value = %f}" % ks_pval_ttVnode
    ks_pval_ttJets = "#scale[0.5]{ttJets node KS p-value = %f}" % ks_pval_ttJetsnode

    latex_ks_test_stat_ttH = ROOT.TLatex(0.7,0.65,ks_test_statistic_ttH)
    latex_ks_pval_ttH = ROOT.TLatex(0.7,0.6,ks_pval_ttH)
    latex_ks_test_stat_ttV = ROOT.TLatex(0.7,0.55,ks_test_statistic_ttV)
    latex_ks_pval_ttV = ROOT.TLatex(0.7,0.5,ks_pval_ttV)
    latex_ks_test_stat_ttJets = ROOT.TLatex(0.7,0.45,ks_test_statistic_ttJets)
    latex_ks_pval_ttJets = ROOT.TLatex(0.7,0.4,ks_pval_ttJets)
    latex_ks_test_stat_ttH.SetNDC()
    latex_ks_pval_ttH.SetNDC()
    latex_ks_test_stat_ttV.SetNDC()
    latex_ks_pval_ttV.SetNDC()
    latex_ks_test_stat_ttJets.SetNDC()
    latex_ks_pval_ttJets.SetNDC()

    latex_ks_test_stat_ttH.Draw("same")
    latex_ks_pval_ttH.Draw("same")
    latex_ks_test_stat_ttV.Draw("same")
    latex_ks_pval_ttV.Draw("same")
    latex_ks_test_stat_ttJets.Draw("same")
    latex_ks_pval_ttJets.Draw("same")

    # Add custom title
    l1=ROOT.TLatex()
    l1.SetNDC();
    l1.DrawLatex(0.36,0.94,"Multiclass DNN Response")

    sep_ttH_v_ttV_testing = GetSeparation(Histo_testing_ttH_DNNResponse, Histo_testing_ttV_DNNResponse)
    sep_ttH_v_ttJ_testing = GetSeparation(Histo_testing_ttH_DNNResponse, Histo_testing_ttJets_DNNResponse)

    '''l2=ROOT.TLatex()
    l2.SetNDC();
    latex_separation_ttH_v_ttV = '#scale[0.5]{ttH vs. ttV separation = %.5f}' % sep_ttH_v_ttV_testing
    l2.DrawLatex(0.7,0.35,latex_separation_ttH_v_ttV)

    l3=ROOT.TLatex()
    l3.SetNDC();
    latex_separation_ttH_v_ttJ = '#scale[0.5]{ttH vs. ttJ separation = %.5f}' % sep_ttH_v_ttJ_testing
    l3.DrawLatex(0.7,0.3,latex_separation_ttH_v_ttJ)'''

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
    ratioframe_ttH.Divide(Histo_testing_ttH_DNNResponse)
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
    classifier_parent_dir = 'MultiClass_DNN_allJets_%s' % (classifier_suffix)
    if node == 'tth':
        histo_DNN_response_ttHsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttH')
        histo_DNN_response_ttVsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttV')
        histo_DNN_response_ttJetssample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttJets')
        histo_DNN_response_ttHsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttH')
        histo_DNN_response_ttVsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttV')
        histo_DNN_response_ttJetssample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttJets')
    elif node == 'ttV':
        histo_DNN_response_ttHsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttH')
        histo_DNN_response_ttVsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttV')
        histo_DNN_response_ttJetssample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttJets')
        histo_DNN_response_ttHsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttH')
        histo_DNN_response_ttVsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttV')
        histo_DNN_response_ttJetssample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttJets')
    elif node == 'ttJets':
        histo_DNN_response_ttHsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttH')
        histo_DNN_response_ttVsample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttV')
        histo_DNN_response_ttJetssample_test = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttJets')
        histo_DNN_response_ttHsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttH')
        histo_DNN_response_ttVsample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttV')
        histo_DNN_response_ttJetssample_train = input_root.Get(classifier_parent_dir+'/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttJets')

    maxyaxis = max(histo_DNN_response_ttHsample_test.GetMaximum(), histo_DNN_response_ttVsample_test.GetMaximum(), histo_DNN_response_ttJetssample_test.GetMaximum(),histo_DNN_response_ttHsample_train.GetMaximum(), histo_DNN_response_ttVsample_train.GetMaximum(), histo_DNN_response_ttJetssample_train.GetMaximum())
    histo_DNN_response_ttHsample_test.SetAxisRange(0.,maxyaxis+1,"Y")

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

    histo_DNN_response_ttHsample_test.GetYaxis().SetTitle("(1/N)dN/dX")

    legend = TLegend(0.8,  0.7,  0.99,  0.99)
    legend.AddEntry(histo_DNN_response_ttHsample_train,"ttH events (train)")
    legend.AddEntry(histo_DNN_response_ttVsample_train,"ttV events (train)")
    legend.AddEntry(histo_DNN_response_ttJetssample_train,"tt+jets events (train)")
    legend.AddEntry(histo_DNN_response_ttHsample_test,"ttH events (test)")
    legend.AddEntry(histo_DNN_response_ttVsample_test,"ttV events (test)")
    legend.AddEntry(histo_DNN_response_ttJetssample_test,"tt+jets events (test)")

    histo_DNN_response_ttHsample_test.Draw('EP')
    histo_DNN_response_ttVsample_test.Draw('EPsame')
    histo_DNN_response_ttJetssample_test.Draw('EPsame')
    histo_DNN_response_ttHsample_train.Draw('HISTsame')
    histo_DNN_response_ttVsample_train.Draw('HISTsame')
    histo_DNN_response_ttJetssample_train.Draw('HISTsame')

    # Add custom title
    l1=ROOT.TLatex()
    l1.SetNDC();
    nodehisto_title = "Multiclass DNN Response: %s" % node
    l1.DrawLatex(0.36,0.94,nodehisto_title)

    if 'tth' in node:
        sep_sig_v_bckg1_testing = GetSeparation(histo_DNN_response_ttHsample_test, histo_DNN_response_ttVsample_test)
        sep_ttH_v_ttJ_testing = GetSeparation(histo_DNN_response_ttHsample_test, histo_DNN_response_ttJetssample_test)
        latex_separation_sig_v_bckg1 = '#scale[0.5]{ttH vs. ttV separation = %.5f}' % sep_sig_v_bckg1_testing
        latex_separation_sig_v_bckg2 = '#scale[0.5]{ttH vs. ttJ separation = %.5f}' % sep_ttH_v_ttJ_testing
    elif 'ttV' in node:
        sep_sig_v_bckg1_testing = GetSeparation(histo_DNN_response_ttVsample_test, histo_DNN_response_ttHsample_test)
        sep_sig_v_bckg2_testing = GetSeparation(histo_DNN_response_ttVsample_test, histo_DNN_response_ttJetssample_test)
        latex_separation_sig_v_bckg1 = '#scale[0.5]{ttV vs. ttH separation = %.5f}' % sep_sig_v_bckg1_testing
        latex_separation_sig_v_bckg2 = '#scale[0.5]{ttV vs. ttJ separation = %.5f}' % sep_sig_v_bckg2_testing
    elif 'ttJ' in node:
        sep_sig_v_bckg1_testing = GetSeparation(histo_DNN_response_ttJetssample_test, histo_DNN_response_ttHsample_test)
        sep_sig_v_bckg2_testing = GetSeparation(histo_DNN_response_ttJetssample_test, histo_DNN_response_ttVsample_test)
        latex_separation_sig_v_bckg1 = '#scale[0.5]{ttJ vs. ttH separation = %.5f}' % sep_sig_v_bckg1_testing
        latex_separation_sig_v_bckg2 = '#scale[0.5]{ttJ vs. ttV separation = %.5f}' % sep_sig_v_bckg2_testing

    l2=ROOT.TLatex()
    l2.SetNDC();
    l2.DrawLatex(0.65,0.55,latex_separation_sig_v_bckg1)

    l3=ROOT.TLatex()
    l3.SetNDC();
    l3.DrawLatex(0.65,0.5,latex_separation_sig_v_bckg2)

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
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify inputs from network training',      default=None,        type='string')
    (opt, args) = parser.parse_args()

    if opt.input_suffix == None:
        print 'Input files suffix not defined!'
        sys.exit(1)

    classifier_suffix = opt.input_suffix

    classifier_parent_dir = 'MultiClass_DNN_allJets_%s' % (classifier_suffix)
    classifier_plots_dir = classifier_parent_dir+"/plots"
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    classifier_samples_dir = classifier_parent_dir+"/outputs"

    input_name = '%s/MultiClass_DNN_allJets_%s.root' % (classifier_samples_dir,classifier_suffix)
    input_root = TFile.Open(input_name)
    traintree_pathname = "MultiClass_DNN_allJets_%s/TrainTree" % (classifier_suffix)
    testtree_pathname = "MultiClass_DNN_allJets_%s/TestTree" % (classifier_suffix)
    # Fetch the trees of events from the root file
    TrainTree = input_root.Get(traintree_pathname)
    TestTree = input_root.Get(testtree_pathname)
    plot_DNNResponse(TrainTree,TestTree, classifier_suffix)
    plot_node_response(input_root,'tth',classifier_suffix)
    plot_node_response(input_root,'ttV',classifier_suffix)
    plot_node_response(input_root,'ttJets',classifier_suffix)

main()
