#################################
#     DNN_ResponsePlotter.py
#     Joshuha Thomas-Wilsker
#       IHEP Bejing, CERN
#################################
# Plotting script using output
# root files from TMVA training.
#################################

import ROOT
from ROOT import TFile, TTree, gDirectory, gROOT, TH1, TF1, TProfile, TProfile2D
#import numpy as np
#import matplotlib.mlab as mlab
#import matplotlib.pyplot as plt


#def plot_ROCS(signal_hist, bckg0_hist, bckg1_hist):
#def plot_DNNResponse(TrainTree, TTree):


def main():

    input_root = TFile.Open('ttHML_MCDNN_hiddenLayers_2.root')
    #input_root.cd('MultiClass_DNN/Method_PyKeras/PyKeras/')

    # Declare and define new hitogram objects
    Histo_training_ttH_DNNResponse = ROOT.TH1D('Histo_training_ttH_DNNResponse','ttH sample (Train)',40,0.0,1.0)
    Histo_training_ttJets_DNNResponse = ROOT.TH1D('Histo_training_ttJets_DNNResponse','ttJets sample (Train)',40,0.0,1.0)
    Histo_training_ttV_DNNResponse = ROOT.TH1D('Histo_training_ttV_DNNResponse','ttV sample (Train)',40,0.0,1.0)

    Histo_testing_ttH_DNNRepsonse = ROOT.TH1D('Histo_testing_ttH_DNNRepsonse','ttH sample (Test)',40,0.0,1.0)
    Histo_testing_ttJets_DNNResponse = ROOT.TH1D('Histo_testing_ttJets_DNNResponse','ttJets sample (Test)',40,0.0,1.0)
    Histo_testing_ttV_DNNResponse = ROOT.TH1D('Histo_testing_ttV_DNNResponse','ttV sample (Test)',40,0.0,1.0)

    # Fetch the trees of events from the root file
    TrainTree = input_root.Get("MultiClass_DNN/TrainTree")
    TestTree = input_root.Get("MultiClass_DNN/TestTree")

    # Now lets project the tree information into those histograms
    # My current understanding is that the three branches beneath
    # represent the response (i.e.)
    TrainTree.Project("Histo_training_ttH_DNNResponse","DNN.sample=ttH")
    TrainTree.Project("Histo_training_ttV_DNNResponse","DNN.sample=ttV")
    TrainTree.Project("Histo_training_ttJets_DNNResponse","DNN.sample=ttJets")

    TestTree.Project("Histo_testing_ttH_DNNRepsonse","DNN.sample=ttH")
    TestTree.Project("Histo_testing_ttV_DNNResponse","DNN.sample=ttV")
    TestTree.Project("Histo_testing_ttJets_DNNResponse","DNN.sample=ttJets")

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
    Histo_training_ttV_DNNResponse.SetLineColor(6)
    Histo_training_ttV_DNNResponse.SetMarkerColor(6)
    Histo_training_ttV_DNNResponse.SetFillColor(6)
    Histo_training_ttV_DNNResponse.SetFillStyle(3001)

    Histo_testing_ttV_DNNResponse.SetLineColor(6)
    Histo_testing_ttV_DNNResponse.SetMarkerColor(6)
    Histo_testing_ttV_DNNResponse.SetFillColor(6)
    Histo_testing_ttV_DNNResponse.SetMarkerStyle(20)

    # Draw the objects
    Histo_training_ttJets_DNNResponse.Draw("HIST")
    Histo_training_ttH_DNNResponse.Draw("HISTSAME")
    Histo_training_ttV_DNNResponse.Draw("HISTSAME")
    Histo_testing_ttH_DNNRepsonse.Draw("EPSAME")
    Histo_testing_ttJets_DNNResponse.Draw("EPSAME")
    Histo_testing_ttV_DNNResponse.Draw("EPSAME")

    #c1.cd(1) : Returns TVirtualPad. Sets current TPad.
    #BuildLegend : Function of TPad Class. Builds legend from graphical objects in pad.
    c1.cd(1).BuildLegend( 0.8,  0.7,  0.99,  0.99,"","LP").SetFillColor(0)

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
    ratioframe_ttV.SetMarkerColor(6)
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
    #outfile_name = 'MCDNN_overtraintest_%s.pdf' % nlayers
    outfile_name = 'MCDNN_overtraintest_2layers.pdf'
    c1.Print(outfile_name,'pdf')

main()
