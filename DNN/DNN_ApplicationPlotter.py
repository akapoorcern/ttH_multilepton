
import ROOT
from ROOT import TMVA, TFile, TString, TLegend, THStack
from array import array
from subprocess import call
from os.path import isfile

def make_plot(histo_0, histo_1, histo_2, suffix, stacked):
    c1 = ROOT.TCanvas('c1',',1000,700')
    p1 = ROOT.TPad('p1','p1',0.0,0.0,1.0,1.0)
    p1.Draw()
    p1.SetRightMargin(0.1)
    p1.SetLeftMargin(0.1)
    p1.SetBottomMargin(0.1)
    p1.SetTopMargin(0.1)
    p1.SetGridx(True)
    p1.SetGridy(True)
    p1.cd()
    #p1.SetLogy(1)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    histo_0.GetYaxis().SetTitle('Counts/Bin')
    histo_0.GetXaxis().SetTitle('DNN Response')

    #maxyaxis = max(histo_0.GetMaximum(), histo_1.GetMaximum(), histo_2.GetMaximum())
    histo_0.SetAxisRange(0.,1000,"Y")
    histo_0.SetLineColor(1)
    histo_0.SetMarkerColor(1)
    #histo_0.SetFillColor(1)
    #histo_0.SetFillStyle(3001)
    histo_0.SetMarkerStyle(20)

    histo_1.SetLineColor(3)
    histo_1.SetMarkerColor(3)
    #histo_1.SetFillColor(3)
    #histo_1.SetFillStyle(3001)
    histo_1.SetMarkerStyle(20)

    histo_2.SetLineColor(4)
    histo_2.SetMarkerColor(4)
    #histo_2.SetFillColor(4)
    #histo_2.SetFillStyle(3001)
    histo_2.SetMarkerStyle(20)

    if stacked == True:
        hs = ROOT.THStack()
        #First on stack goes on bottom.
        hs.Add(histo_2)
        hs.Add(histo_1)
        hs.Draw()
        hs.GetYaxis().SetTitle('Counts/Bin')
        hs.GetXaxis().SetTitle('DNN Response')
        histo_0.Draw("same")
        c1.Update()
        legend = TLegend(0.13,  0.7,  0.4,  0.90)
        legend.AddEntry(histo_0,"Assigned ttH events")
        legend.AddEntry(histo_1,"Assigned ttV events")
        legend.AddEntry(histo_2,"Assigned tt+jets events")
        legend.Draw("same")
        outfile_name = 'MCDNN_trainingRegion_stacked_%s.pdf'%suffix
        c1.Print(outfile_name,'pdf')
    elif stacked == False:
        histo_0.Draw("HIST")
        histo_1.Draw("HISTSAME")
        histo_2.Draw("HISTSAME")
        legend = TLegend(0.23,  0.7,  0.4,  0.90)
        legend.AddEntry(histo_0,"Assigned ttH events")
        legend.AddEntry(histo_1,"Assigned ttV events")
        legend.AddEntry(histo_2,"Assigned tt+jets events")
        legend.Draw("same")
        outfile_name = 'MCDNN_trainingRegion_%s.pdf'%suffix
        c1.Print(outfile_name,'pdf')

def main():
    input_file = TFile.Open('ttHML_MCDNN_applied-new.root')

    histo_ttHsample_ttHnode = input_file.Get('histo_ttHsample_ttHnode')
    histo_ttVsample_ttHnode = input_file.Get('histo_ttVsample_ttHnode')
    histo_ttJetssample_ttHnode = input_file.Get('histo_ttJetssample_ttHnode')

    histo_ttVsample_ttVnode = input_file.Get('histo_ttVsample_ttVnode')
    histo_ttJetssample_ttJetsnode = input_file.Get('histo_ttJetssample_ttJetsnode')
    histo_ttHsample_ttHnode_ttHassigned = input_file.Get('histo_ttHsample_ttHnode_ttHassigned')
    histo_ttVsample_ttHnode_ttHassigned = input_file.Get('histo_ttVsample_ttHnode_ttHassigned')
    histo_ttJetssample_ttHnode_ttHassigned = input_file.Get('histo_ttJetssample_ttHnode_ttHassigned')
    histo_ttHsample_ttVnode_ttVassigned = input_file.Get('histo_ttHsample_ttVnode_ttVassigned')
    histo_ttVsample_ttVnode_ttVassigned = input_file.Get('histo_ttVsample_ttVnode_ttVassigned')
    histo_ttJetssample_ttVnode_ttVassigned = input_file.Get('histo_ttJetssample_ttVnode_ttVassigned')
    histo_ttHsample_ttJetsnode_ttJetsassigned = input_file.Get('histo_ttHsample_ttJetsnode_ttJetsassigned')
    histo_ttVsample_ttJetsnode_ttJetsassigned = input_file.Get('histo_ttVsample_ttJetsnode_ttJetsassigned')
    histo_ttJetssample_ttJetsnode_ttJetsassigned = input_file.Get('histo_ttJetssample_ttJetsnode_ttJetsassigned')

    make_plot(histo_ttHsample_ttHnode,histo_ttVsample_ttVnode,histo_ttJetssample_ttJetsnode, 'processFromNode-new', False)
    make_plot(histo_ttHsample_ttHnode_ttHassigned, histo_ttVsample_ttHnode_ttHassigned, histo_ttJetssample_ttHnode_ttHassigned, 'ttHassigned-new', False)
    make_plot(histo_ttHsample_ttVnode_ttVassigned, histo_ttVsample_ttVnode_ttVassigned, histo_ttJetssample_ttVnode_ttVassigned, 'ttVassigned-new', False)
    make_plot(histo_ttHsample_ttJetsnode_ttJetsassigned, histo_ttVsample_ttJetsnode_ttJetsassigned, histo_ttJetssample_ttJetsnode_ttJetsassigned, 'ttJetassigned-new', False)

    make_plot(histo_ttHsample_ttHnode, histo_ttVsample_ttHnode, histo_ttJetssample_ttHnode, 'ttHnode', False)

main()
