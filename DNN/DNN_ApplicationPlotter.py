
import ROOT, os, optparse
from ROOT import TMVA, TFile, TString, TLegend, THStack
from array import array
from subprocess import call
from os.path import isfile

def make_plot(histo_0, histo_1, histo_2, suffix, stacked, classifier_parent_dir):
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

    histo_0.GetYaxis().SetTitle('Counts/Bin (normalised)')
    histo_0.GetXaxis().SetTitle('DNN Response')

    data_ttH = TFile.Open('samples/DiLepTR_ttH_bInclude.root')
    data_ttV = TFile.Open('samples/DiLepTR_ttV_bInclude.root')
    data_ttJets = TFile.Open('samples/DiLepTR_ttJets_bInclude.root')
    data_ttH_tree = data_ttH.Get('BOOM')
    data_ttV_tree = data_ttV.Get('BOOM')
    data_ttJets_tree = data_ttJets.Get('BOOM')
    total_ttH_events = data_ttH_tree.GetEntries()
    total_ttV_events = data_ttV_tree.GetEntries()
    total_ttJ_events = data_ttJets_tree.GetEntries()

    dx_ttH = (histo_0.GetXaxis().GetXmax() - histo_0.GetXaxis().GetXmin()) / histo_0.GetNbinsX()
    ttH_norm = 1./histo_0.GetSumOfWeights()/dx_ttH
    histo_0.Scale(ttH_norm)

    dx_ttV = (histo_1.GetXaxis().GetXmax() - histo_1.GetXaxis().GetXmin()) / histo_1.GetNbinsX()
    ttV_norm = 1./histo_1.GetSumOfWeights()/dx_ttV
    histo_1.Scale(ttV_norm)

    dx_ttJ = (histo_2.GetXaxis().GetXmax() - histo_2.GetXaxis().GetXmin()) / histo_2.GetNbinsX()
    ttJ_norm = 1./histo_2.GetSumOfWeights()/dx_ttJ
    histo_2.Scale(ttJ_norm)

    maxyaxis = max(histo_0.GetMaximum(), histo_1.GetMaximum(), histo_2.GetMaximum())
    histo_0.SetAxisRange(0.,maxyaxis+1.,"Y")
    histo_0.SetLineColor(2)
    histo_0.SetMarkerColor(2)
    histo_0.SetMarkerStyle(20)
    histo_0.SetFillColor(2)
    histo_0.SetFillStyle(3001)

    histo_1.SetLineColor(3)
    histo_1.SetMarkerColor(3)
    histo_1.SetMarkerStyle(20)
    histo_1.SetFillColor(3)
    histo_1.SetFillStyle(3001)

    histo_2.SetLineColor(4)
    histo_2.SetMarkerColor(4)
    histo_2.SetMarkerStyle(20)
    histo_2.SetFillColor(4)
    histo_2.SetFillStyle(3001)

    if stacked == True:
        hs = ROOT.THStack()
        #First on stack goes on bottom.
        hs.Add(histo_2)
        hs.Add(histo_1)
        hs.Draw()
        hs.GetYaxis().SetTitle('Counts/Bin (normalised)')
        hs.GetXaxis().SetTitle('DNN Response')
        histo_0.Draw("same")
        c1.Update()
        legend = TLegend(0.85,  0.85,  1.,  1.)
        legend.AddEntry(histo_0,"ttH node ttH sample")
        legend.AddEntry(histo_1,"ttV node ttH sample")
        legend.AddEntry(histo_2,"ttJets node ttH sample")
        legend.Draw("same")
        outfile_name = 'MCDNN_Response_Applied_stacked_%s.pdf'%suffix
        c1.Print(outfile_name,'pdf')
    elif stacked == False:
        histo_0.Draw("HIST")
        histo_1.Draw("HISTSAME")
        histo_2.Draw("HISTSAME")
        legend = TLegend(0.85,  0.85,  1.,  1.)
        legend.AddEntry(histo_0,"ttH node ttH sample")
        legend.AddEntry(histo_1,"ttV node ttH sample")
        legend.AddEntry(histo_2,"ttJets node ttH sample")
        legend.Draw("same")

        classifier_plots_dir = classifier_parent_dir+"/plots"
        if not os.path.exists(classifier_plots_dir):
            os.makedirs(classifier_plots_dir)

        # Add custom title
        l1=ROOT.TLatex()
        l1.SetNDC();
        latex_title = "Multiclass DNN Response: %s" % suffix
        l1.DrawLatex(0.2,0.94,latex_title)

        outfile_name = 'MCDNN_Response_Applied_%s.pdf'%suffix
        output_fullpath = classifier_plots_dir + '/' + outfile_name
        c1.Print(output_fullpath,'pdf')

def main():

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify inputs from network training',      default=None,        type='string')
    (opt, args) = parser.parse_args()

    if opt.input_suffix == None:
        print 'Input files suffix not defined!'
        sys.exit(1)

    classifier_suffix = opt.input_suffix
    classifier_parent_dir = 'MultiClass_DNN_%s' % (classifier_suffix)
    classifier_samples_dir = classifier_parent_dir+"/outputs"
    input_name = '%s/Applied_%s.root' % (classifier_samples_dir,classifier_parent_dir)
    print 'input_name: ', input_name
    input_file = TFile.Open(input_name)
    input_file.ls()

    histo_ttHnode_ttHsample = input_file.Get('histo_ttHnode_ttH')
    histo_ttHnode_ttVsample = input_file.Get('histo_ttHnode_ttV')
    histo_ttHnode_ttJetssample = input_file.Get('histo_ttHnode_ttJets')

    histo_ttVnode_ttHsample = input_file.Get('histo_ttVnode_ttH')
    histo_ttVnode_ttVsample = input_file.Get('histo_ttVnode_ttV')
    histo_ttVnode_ttJetssample = input_file.Get('histo_ttVnode_ttJets')

    histo_ttJetsnode_ttHsample = input_file.Get('histo_ttJetsnode_ttH')
    histo_ttJetsnode_ttVsample = input_file.Get('histo_ttJetsnode_ttV')
    histo_ttJetsnode_ttJetssample = input_file.Get('histo_ttJetsnode_ttJets')

    histo_ttHclassified_ttHsample = input_file.Get('histo_ttHclassified_events_ttH')
    histo_ttHclassified_ttVsample = input_file.Get('histo_ttHclassified_events_ttV')
    histo_ttHclassified_ttJsample = input_file.Get('histo_ttHclassified_events_ttJets')

    histo_ttVclassified_ttHsample = input_file.Get('histo_ttVclassified_events_ttH')
    histo_ttVclassified_ttVsample = input_file.Get('histo_ttVclassified_events_ttV')
    histo_ttVclassified_ttJsample = input_file.Get('histo_ttVclassified_events_ttJets')

    histo_ttJclassified_ttHsample = input_file.Get('histo_ttJclassified_events_ttH')
    histo_ttJclassified_ttVsample = input_file.Get('histo_ttJclassified_events_ttV')
    histo_ttJclassified_ttJsample = input_file.Get('histo_ttJclassified_events_ttJets')

    make_plot(histo_ttHnode_ttHsample,histo_ttHnode_ttVsample,histo_ttHnode_ttJetssample, 'ttH_node', False, classifier_parent_dir)
    make_plot(histo_ttVnode_ttHsample,histo_ttVnode_ttVsample,histo_ttVnode_ttJetssample, 'ttV_node', False, classifier_parent_dir)
    make_plot(histo_ttJetsnode_ttHsample,histo_ttJetsnode_ttVsample,histo_ttJetsnode_ttJetssample, 'ttJets_node', False, classifier_parent_dir)
    make_plot(histo_ttHclassified_ttHsample,histo_ttHclassified_ttVsample,histo_ttHclassified_ttJsample, 'ttH_category', False, classifier_parent_dir)
    make_plot(histo_ttVclassified_ttHsample,histo_ttVclassified_ttVsample,histo_ttVclassified_ttJsample, 'ttV_category', False, classifier_parent_dir)
    make_plot(histo_ttJclassified_ttHsample,histo_ttJclassified_ttVsample,histo_ttJclassified_ttJsample, 'ttJ_category', False, classifier_parent_dir)

main()
