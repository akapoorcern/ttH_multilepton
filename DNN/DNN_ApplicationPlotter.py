
import ROOT, os, optparse
from ROOT import TMVA, TFile, TString, TLegend, THStack, TLatex
from array import array
from subprocess import call
from os.path import isfile

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
    p1.SetLogy(1)
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

    '''
    dx_ttH = (histo_0.GetXaxis().GetXmax() - histo_0.GetXaxis().GetXmin()) / histo_0.GetNbinsX()
    ttH_norm = 1./histo_0.GetSumOfWeights()/dx_ttH
    histo_0.Scale(ttH_norm)
    dx_ttV = (histo_1.GetXaxis().GetXmax() - histo_1.GetXaxis().GetXmin()) / histo_1.GetNbinsX()
    ttV_norm = 1./histo_1.GetSumOfWeights()/dx_ttV
    histo_1.Scale(ttV_norm)
    dx_ttJ = (histo_2.GetXaxis().GetXmax() - histo_2.GetXaxis().GetXmin()) / histo_2.GetNbinsX()
    ttJ_norm = 1./histo_2.GetSumOfWeights()/dx_ttJ
    histo_2.Scale(ttJ_norm)
    '''

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

    classifier_plots_dir = classifier_parent_dir+"/plots"
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    if stacked == True:
        hs = ROOT.THStack()
        #First on stack goes on bottom.
        hs.Add(histo_1)
        hs.Add(histo_2)
        hs.Add(histo_0)
        hs.Draw()
        hs.GetYaxis().SetTitle('Counts/Bin')
        hs.GetXaxis().SetTitle('DNN Response')
        print 'Sum of weights (stacked): ', hs.GetStack().Last().GetSumOfWeights()
        legend = TLegend(0.85,  0.85,  1.,  1.)
        legend.AddEntry(histo_0,"ttH node ttH sample")
        legend.AddEntry(histo_1,"ttV node ttH sample")
        legend.AddEntry(histo_2,"ttJets node ttH sample")
        legend.Draw("same")
        histo_0.SetFillStyle(0)
        histo_0.SetLineColor(1)
        signal_scale = hs.GetStack().Last().GetSumOfWeights()/histo_0.GetSumOfWeights()
        print 'signal_scale: ', signal_scale
        histo_0.Scale(signal_scale)
        #histo_0.Draw("HISTsame")
        #c1.SetLogy()
        c1.Update()
        outfile_name = 'MCDNN_Response_Applied_stacked_%s.pdf'%suffix
        output_fullpath = classifier_plots_dir + '/' + outfile_name
        c1.Print(output_fullpath,'pdf')
    elif stacked == False:
        histo_0.Draw("HIST")
        histo_1.Draw("HISTSAME")
        histo_2.Draw("HISTSAME")
        legend = TLegend(0.85,  0.85,  1.,  1.)
        legend.AddEntry(histo_0,"ttH node ttH sample")
        legend.AddEntry(histo_1,"ttV node ttH sample")
        legend.AddEntry(histo_2,"ttJets node ttH sample")
        legend.Draw("same")

        # Add custom title
        l1=ROOT.TLatex()
        l1.SetNDC();
        latex_title = "Multiclass DNN Response: %s" % suffix
        l1.DrawLatex(0.2,0.94,latex_title)

        if 'ttH_category' in suffix:
            sep_sig_v_bckg1_testing = GetSeparation(histo_0, histo_1)
            sep_sig_v_bckg2_testing = GetSeparation(histo_0, histo_2)
            l2=ROOT.TLatex()
            l2.SetNDC();
            latex_separation_sig_v_bckg1 = '#scale[0.5]{ttH vs. ttV separation = %.4f}' % sep_sig_v_bckg1_testing
            l2.DrawLatex(0.6,0.8,latex_separation_sig_v_bckg1)
            l3=ROOT.TLatex()
            l3.SetNDC();
            latex_separation_sig_v_bckg2 = '#scale[0.5]{ttH vs. ttJ separation = %.4f}' % sep_sig_v_bckg2_testing
            l3.DrawLatex(0.6,0.75,latex_separation_sig_v_bckg2)

        if 'ttV_category' in suffix:
            sep_sig_v_bckg1_testing = GetSeparation(histo_1, histo_0)
            sep_sig_v_bckg2_testing = GetSeparation(histo_1, histo_2)
            l2=ROOT.TLatex()
            l2.SetNDC();
            latex_separation_sig_v_bckg1 = '#scale[0.5]{ttV vs. ttH separation = %.4f}' % sep_sig_v_bckg1_testing
            l2.DrawLatex(0.6,0.8,latex_separation_sig_v_bckg1)
            l3=ROOT.TLatex()
            l3.SetNDC();
            latex_separation_sig_v_bckg2 = '#scale[0.5]{ttV vs. ttJ separation = %.4f}' % sep_sig_v_bckg2_testing
            l3.DrawLatex(0.6,0.75,latex_separation_sig_v_bckg2)

        if 'ttJ_category' in suffix:
            sep_sig_v_bckg1_testing = GetSeparation(histo_2, histo_0)
            sep_sig_v_bckg2_testing = GetSeparation(histo_2, histo_1)
            l2=ROOT.TLatex()
            l2.SetNDC();
            latex_separation_sig_v_bckg1 = '#scale[0.5]{ttJ vs. ttH separation = %.4f}' % sep_sig_v_bckg1_testing
            l2.DrawLatex(0.6,0.8,latex_separation_sig_v_bckg1)
            l3=ROOT.TLatex()
            l3.SetNDC();
            latex_separation_sig_v_bckg2 = '#scale[0.5]{ttJ vs. ttV separation = %.4f}' % sep_sig_v_bckg2_testing
            l3.DrawLatex(0.6,0.75,latex_separation_sig_v_bckg2)

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

    #histo_ttHnode_ttHsample = input_file.Get('histo_ttHnode_ttH')
    #histo_ttHnode_ttVsample = input_file.Get('histo_ttHnode_ttV')
    #histo_ttHnode_ttJetssample = input_file.Get('histo_ttHnode_ttJets')

    #histo_ttVnode_ttHsample = input_file.Get('histo_ttVnode_ttH')
    #histo_ttVnode_ttVsample = input_file.Get('histo_ttVnode_ttV')
    #histo_ttVnode_ttJetssample = input_file.Get('histo_ttVnode_ttJets')

    #histo_ttJetsnode_ttHsample = input_file.Get('histo_ttJetsnode_ttH')
    #histo_ttJetsnode_ttVsample = input_file.Get('histo_ttJetsnode_ttV')
    #histo_ttJetsnode_ttJetssample = input_file.Get('histo_ttJetsnode_ttJets')

    histo_ttHclassified_ttHsample = input_file.Get('histo_ttHclassified_events_ttH')
    histo_ttHclassified_ttVsample = input_file.Get('histo_ttHclassified_events_ttV')
    histo_ttHclassified_ttJsample = input_file.Get('histo_ttHclassified_events_ttJets')

    histo_ttVclassified_ttHsample = input_file.Get('histo_ttVclassified_events_ttH')
    histo_ttVclassified_ttVsample = input_file.Get('histo_ttVclassified_events_ttV')
    histo_ttVclassified_ttJsample = input_file.Get('histo_ttVclassified_events_ttJets')

    histo_ttJclassified_ttHsample = input_file.Get('histo_ttJclassified_events_ttH')
    histo_ttJclassified_ttVsample = input_file.Get('histo_ttJclassified_events_ttV')
    histo_ttJclassified_ttJsample = input_file.Get('histo_ttJclassified_events_ttJets')

    #make_plot(histo_ttHnode_ttHsample,histo_ttHnode_ttVsample,histo_ttHnode_ttJetssample, 'ttH_node', False, classifier_parent_dir)
    #make_plot(histo_ttVnode_ttHsample,histo_ttVnode_ttVsample,histo_ttVnode_ttJetssample, 'ttV_node', False, classifier_parent_dir)
    #make_plot(histo_ttJetsnode_ttHsample,histo_ttJetsnode_ttVsample,histo_ttJetsnode_ttJetssample, 'ttJets_node', False, classifier_parent_dir)
    make_plot(histo_ttHclassified_ttHsample,histo_ttHclassified_ttVsample,histo_ttHclassified_ttJsample, 'ttH_category', True, classifier_parent_dir)
    make_plot(histo_ttVclassified_ttHsample,histo_ttVclassified_ttVsample,histo_ttVclassified_ttJsample, 'ttV_category', True, classifier_parent_dir)
    make_plot(histo_ttJclassified_ttHsample,histo_ttJclassified_ttVsample,histo_ttJclassified_ttJsample, 'ttJ_category', True, classifier_parent_dir)

main()
