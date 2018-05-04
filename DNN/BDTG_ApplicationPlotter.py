
import ROOT, os, optparse
from ROOT import TMVA, TFile, TString, TLegend, THStack
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

def make_plot(histo_0, histo_1, suffix, stacked, classifier_parent_dir):
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
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    histo_0.GetYaxis().SetTitle('(1/N)dN/dX')
    histo_0.GetXaxis().SetTitle('Response')

    dx_ttH = (histo_0.GetXaxis().GetXmax() - histo_0.GetXaxis().GetXmin()) / histo_0.GetNbinsX()
    ttH_norm = 1./histo_0.GetSumOfWeights()/dx_ttH
    histo_0.Scale(ttH_norm)

    dx_bckg = (histo_1.GetXaxis().GetXmax() - histo_1.GetXaxis().GetXmin()) / histo_1.GetNbinsX()
    bckg_norm = 1./histo_1.GetSumOfWeights()/dx_bckg
    histo_1.Scale(bckg_norm)

    sig_bckg_sep = GetSeparation(histo_0, histo_1)

    maxyaxis = max(histo_0.GetMaximum(), histo_1.GetMaximum())
    histo_0.SetAxisRange(0.,maxyaxis+1.,"Y")
    histo_0.SetLineColor(2)
    histo_0.SetMarkerColor(2)
    histo_0.SetMarkerStyle(20)
    histo_0.SetFillColor(2)
    histo_0.SetFillStyle(3001)

    if 'ttV' in suffix:
        color = 3
    elif 'ttJets' in suffix:
        color = 4
    histo_1.SetLineColor(color)
    histo_1.SetMarkerColor(color)
    histo_1.SetMarkerStyle(20)
    histo_1.SetFillColor(color)
    histo_1.SetFillStyle(3001)

    histo_0.Draw("HIST")
    histo_1.Draw("HISTSAME")
    legend = TLegend(0.85,  0.85,  1.,  1.)
    legend.AddEntry(histo_0,"ttH")
    if 'ttV' in suffix:
        legend.AddEntry(histo_1,"ttV")
    if 'ttJets' in suffix:
        legend.AddEntry(histo_1,"ttJets")
    legend.Draw("same")

    classifier_plots_dir = classifier_parent_dir+"/plots"
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    # Add custom title
    l1=ROOT.TLatex()
    l1.SetNDC()
    latex_title = "BDTG Response: %s" % suffix
    l1.DrawLatex(0.25,0.94,latex_title)

    l2=ROOT.TLatex()
    l2.SetNDC()
    latex_separation = '#scale[0.5]{Separation = %.5f}' % sig_bckg_sep
    l2.DrawLatex(0.7,0.8,latex_separation)

    outfile_name = 'BDTG_Response_Applied_%s.pdf'%suffix
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
    classifier_parent_dir = 'BinaryClassifier_BDTG_%s' % (classifier_suffix)
    classifier_samples_dir = classifier_parent_dir+"/outputs"
    input_name = '%s/Applied_%s.root' % (classifier_samples_dir,classifier_parent_dir)
    print 'input_name: ', input_name
    input_file = TFile.Open(input_name)
    input_file.ls()

    histo_response_ttH = input_file.Get('histo_response_ttH')
    if classifier_suffix == 'ttHvsttV':
        histo_response_bckg = input_file.Get('histo_response_ttV')
    elif classifier_suffix == 'ttHvsttJets':
        histo_response_bckg = input_file.Get('histo_response_ttjets')

    make_plot(histo_response_ttH,histo_response_bckg, classifier_suffix, False, classifier_parent_dir)

main()
