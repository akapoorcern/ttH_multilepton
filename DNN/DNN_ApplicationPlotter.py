
import ROOT, os, optparse
from ROOT import TMVA, TFile, TString, TLegend, THStack, TLatex, TH1D
from array import array
from subprocess import call
from os.path import isfile
import json

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

    for i in xrange(1,nBins):
        sig_bin_norm = hist_sig.GetBinContent(i)/nS
        bckg_bin_norm = hist_bckg.GetBinContent(i)/nB
        # Separation:
        if(sig_bin_norm+bckg_bin_norm > 0):
            separation += 0.5 * ((sig_bin_norm - bckg_bin_norm) * (sig_bin_norm - bckg_bin_norm)) / (sig_bin_norm + bckg_bin_norm)
    separation *= dX
    return separation

# Create signal/background histogram where signal is just the first histogram passed to method and background = sum of second and third histograms passed to method. Returns S/B histogram only.
def GetSignalOverBackground(hist_sig, hist_bckg1, hist_bckg2):
    nBins = hist_sig.GetNbinsX()
    x_bin_edges = []
    x_bin_contents = []
    x_bin_edges.append(0.)
    for bin_index in xrange(1,nBins):
        nentries_sig = hist_sig.GetBinContent(bin_index)
        nentries_bckg1 = hist_bckg1.GetBinContent(bin_index)
        nentries_bckg2 = hist_bckg2.GetBinContent(bin_index)
        hist_weight = nentries_sig/(nentries_bckg1+nentries_bckg2)
        new_x_bin_edge = hist_sig.GetXaxis().GetBinUpEdge(bin_index)
        x_bin_edges.append(new_x_bin_edge)
        x_bin_contents.append(hist_weight)
    x_bin_edges.append(1.)

    n_xbins = len(x_bin_edges)-1
    x_bin_edges_fuckingroot = array('d',x_bin_edges)
    sOverB = ROOT.TH1D('sOverB','Signal / Total Background', n_xbins, x_bin_edges_fuckingroot)
    for bin_index in xrange(1,nBins):
        sOverB.SetBinContent(bin_index, x_bin_contents[bin_index-1])
    return sOverB

# Rebin histograms so at least one background event (from either backgrounds) exists in each bin. Return vector of histograms (returned in same order as passed to function).
def rebinHistograms(original_hist_sig, original_hist_bckg1, original_hist_bckg2):
    nBins = original_hist_sig.GetNbinsX()
    x_bin_edges = []
    x_bin_edges.append(0.)
    x_axis_bckg1 = original_hist_bckg1.GetXaxis()
    cumulative_sum_entries = 0
    for x_bin_index in xrange(1,nBins):
        nentries_sig = original_hist_sig.GetBinContent(x_bin_index)
        nentries_bckg1 = original_hist_bckg1.GetBinContent(x_bin_index)
        nentries_bckg2 = original_hist_bckg2.GetBinContent(x_bin_index)
        cumulative_sum_entries = cumulative_sum_entries + nentries_bckg1 + nentries_bckg2
        if cumulative_sum_entries > 0:
            new_x_bin_edge = x_axis_bckg1.GetBinUpEdge(x_bin_index)
            x_bin_edges.append(new_x_bin_edge)
            cumulative_sum_entries = 0
    x_bin_edges.append(1.)
    n_xbins = len(x_bin_edges)-1
    x_bin_edges_fuckingroot = array('d',x_bin_edges)
    HistSig_rebin = original_hist_sig.Rebin(n_xbins,'HistSig_rebin',x_bin_edges_fuckingroot)
    HistBckg1_rebin = original_hist_bckg1.Rebin(n_xbins,'HistBckg1_rebin',x_bin_edges_fuckingroot)
    HistBckg2_rebin = original_hist_bckg2.Rebin(n_xbins,'HistBckg2_rebin',x_bin_edges_fuckingroot)
    new_hists = [HistSig_rebin, HistBckg1_rebin, HistBckg2_rebin]
    return new_hists

# Makes plots of the various categories for the DNN. Each 'category' histogram is a plot of the response on one of the DNN nodes.
# However, it contains only the events that showed the highest DNN response for said node.
def make_plot(histo_0, histo_1, histo_2, suffix, stacked, classifier_parent_dir, signalOverBckg, norm):

    c1 = ROOT.TCanvas('c1',',1000,800')
    p1 = ROOT.TPad('p1','p1',0.0,0.2,1.0,1.0)
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


    data_ttH = TFile.Open('samples/DiLepTR_ttH_bInclude.root')
    data_ttV = TFile.Open('samples/DiLepTR_ttV_bInclude.root')
    data_ttJets = TFile.Open('samples/DiLepTR_ttJets_bInclude.root')
    data_ttH_tree = data_ttH.Get('BOOM')
    data_ttV_tree = data_ttV.Get('BOOM')
    data_ttJets_tree = data_ttJets.Get('BOOM')
    total_ttH_events = data_ttH_tree.GetEntries()
    total_ttV_events = data_ttV_tree.GetEntries()
    total_ttJ_events = data_ttJets_tree.GetEntries()

    signal_colour = 0
    bckg1_colour = 0
    bckg2_colour = 0
    legend = TLegend(0.85,  0.85,  1.,  1.)
    if suffix == 'ttH_category':
        signal_colour = 2
        bckg1_colour = 3
        bckg2_colour = 4
        legend.AddEntry(histo_0,"ttH sample")
        legend.AddEntry(histo_1,"ttV sample")
        legend.AddEntry(histo_2,"ttJets sample")
        # Add custom title
        l1=ROOT.TLatex()
        l1.SetNDC();
        latex_title = "ttH Category"
    if suffix == 'ttV_category':
        signal_colour = 3
        bckg1_colour = 2
        bckg2_colour = 4
        legend.AddEntry(histo_0,"ttV sample")
        legend.AddEntry(histo_1,"ttH sample")
        legend.AddEntry(histo_2,"ttJets sample")
        # Add custom title
        l1=ROOT.TLatex()
        l1.SetNDC();
        latex_title = "ttV Category"
    if suffix == 'ttJ_category':
        signal_colour = 4
        bckg1_colour = 2
        bckg2_colour = 3
        legend.AddEntry(histo_0,"ttJets sample")
        legend.AddEntry(histo_1,"ttH sample")
        legend.AddEntry(histo_2,"ttV sample")
        # Add custom title
        l1=ROOT.TLatex()
        l1.SetNDC();
        latex_title = "tt+Jets Category"

    norm_suffix = ''
    if norm == True:
        histo_0.GetYaxis().SetTitle("(1/N)dN/dX")
        histo_1.GetYaxis().SetTitle("(1/N)dN/dX")
        histo_2.GetYaxis().SetTitle("(1/N)dN/dX")
        norm_suffix = 'Norm'
        dx_hist0_train = (histo_0.GetXaxis().GetXmax() - histo_0.GetXaxis().GetXmin()) / histo_0.GetNbinsX()
        hist0_train_norm = 1./histo_0.GetSumOfWeights()/dx_hist0_train
        histo_0.Scale(hist0_train_norm)

        dx_hist1_train = (histo_1.GetXaxis().GetXmax() - histo_1.GetXaxis().GetXmin()) / histo_1.GetNbinsX()
        hist1_train_norm = 1./histo_1.GetSumOfWeights()/dx_hist1_train
        histo_1.Scale(hist1_train_norm)

        dx_hist2_train = (histo_2.GetXaxis().GetXmax() - histo_2.GetXaxis().GetXmin()) / histo_2.GetNbinsX()
        hist2_train_norm = 1./histo_2.GetSumOfWeights()/dx_hist2_train
        histo_2.Scale(hist2_train_norm)
    else:
        histo_0.GetYaxis().SetTitle("Events/bin")
        norm_suffix = 'noNorm'

    maxyaxis = max(histo_0.GetMaximum(), histo_1.GetMaximum(), histo_2.GetMaximum())


    histo_0.SetAxisRange(0.,maxyaxis+1.,"Y")
    histo_1.SetAxisRange(0.,maxyaxis+1.,"Y")
    histo_2.SetAxisRange(0.,maxyaxis+1.,"Y")

    histo_0.SetLineColor(signal_colour)
    histo_0.SetMarkerColor(signal_colour)
    histo_0.SetMarkerStyle(20)
    histo_0.SetFillColor(signal_colour)
    histo_0.SetFillStyle(3001)

    histo_1.SetLineColor(bckg1_colour)
    histo_1.SetMarkerColor(bckg1_colour)
    histo_1.SetMarkerStyle(20)
    histo_1.SetFillColor(bckg1_colour)
    histo_1.SetFillStyle(3001)

    histo_2.SetLineColor(bckg2_colour)
    histo_2.SetMarkerColor(bckg2_colour)
    histo_2.SetMarkerStyle(20)
    histo_2.SetFillColor(bckg2_colour)
    histo_2.SetFillStyle(3001)

    classifier_plots_dir = classifier_parent_dir+"/plots"
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    if stacked == True:
        hs = ROOT.THStack()
        #First on stack goes on bottom.
        if suffix == 'ttH_category':
            hs.Add(histo_2)
            hs.Add(histo_1)
            hs.Add(histo_0)
        if suffix == 'ttV_category':
            hs.Add(histo_2)
            hs.Add(histo_0)
            hs.Add(histo_1)
        if suffix == 'ttJ_category':
            hs.Add(histo_0)
            hs.Add(histo_2)
            hs.Add(histo_1)

        hs.Draw("HIST")
        hs.GetYaxis().SetTitle('Counts/Bin')
        hs.GetXaxis().SetTitle('DNN Response')
        legend.Draw("same")
        l1.DrawLatex(0.4,0.94,latex_title)

        # Draw Signal/Total Bckg ratio plot
        c1.cd()
        p2 = ROOT.TPad('p2','p2',0.0,0.0,1.0,0.2)
        p2.Draw()
        p2.SetLeftMargin(0.1)
        p2.SetRightMargin(0.1)
        p2.SetTopMargin(0.05)
        p2.SetBottomMargin(0.4)
        p2.SetGridx(True)
        p2.SetGridy(True)
        p2.cd()
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptTitle(0)
        signalOverBckg_maxY = signalOverBckg.GetMaximum()

        signalOverBckg.GetYaxis().SetTitle('S/B')
        signalOverBckg.GetYaxis().SetRangeUser(0.,signalOverBckg_maxY+(signalOverBckg_maxY*0.1))
        signalOverBckg.GetYaxis().SetNdivisions(6)
        signalOverBckg.GetYaxis().SetLabelSize(0.12)
        signalOverBckg.GetYaxis().SetTitleSize(0.12)
        signalOverBckg.GetYaxis().SetTitleOffset(0.2)
        signalOverBckg.GetXaxis().SetTitle('Response')
        signalOverBckg.GetXaxis().SetLabelSize(0.15)
        signalOverBckg.GetXaxis().SetTitleSize(0.15)
        signalOverBckg.GetXaxis().SetTitleOffset(1.)
        signalOverBckg.SetFillStyle(0)
        signalOverBckg.SetMarkerStyle(2)
        signalOverBckg.SetMarkerColor(1)
        signalOverBckg.Draw('P')
        c1.Update()
        outfile_name = 'MCDNN_Response_Applied_stacked_%s_%s.pdf'%(suffix,norm_suffix)
        output_fullpath = classifier_plots_dir + '/' + outfile_name
        c1.Print(output_fullpath,'pdf')
    elif stacked == False:
        if suffix == 'ttH_category':
            histo_0.SetFillStyle(0)
            histo_1.SetFillStyle(0)
            histo_2.SetFillStyle(0)
            histo_1.Draw("HIST")
            histo_2.Draw("HISTSAME")
            histo_0.Draw("HISTSAME")
        if suffix == 'ttV_category':
            histo_0.SetFillStyle(0)
            histo_1.SetFillStyle(0)
            histo_2.SetFillStyle(0)
            histo_0.Draw("HIST")
            histo_2.Draw("HISTSAME")
            histo_1.Draw("HISTSAME")
        if suffix == 'ttJ_category':
            histo_0.SetFillStyle(0)
            histo_1.SetFillStyle(0)
            histo_2.SetFillStyle(0)
            histo_0.Draw("HIST")
            histo_2.Draw("HISTSAME")
            histo_1.Draw("HISTSAME")

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

        # Draw Signal/Total Bckg ratio plot
        c1.cd()
        p2 = ROOT.TPad('p2','p2',0.0,0.0,1.0,0.2)
        p2.Draw()
        p2.SetLeftMargin(0.1)
        p2.SetRightMargin(0.1)
        p2.SetTopMargin(0.05)
        p2.SetBottomMargin(0.4)
        p2.SetGridx(True)
        p2.SetGridy(True)
        p2.cd()
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptTitle(0)

        signalOverBckg_maxY = signalOverBckg.GetMaximum()
        signalOverBckg.GetYaxis().SetTitle('S/B')
        signalOverBckg.GetYaxis().SetRangeUser(0.,signalOverBckg_maxY+(signalOverBckg_maxY*0.1))
        signalOverBckg.GetYaxis().SetNdivisions(6)
        signalOverBckg.GetYaxis().SetLabelSize(0.12)
        signalOverBckg.GetYaxis().SetTitleSize(0.12)
        signalOverBckg.GetYaxis().SetTitleOffset(0.2)
        signalOverBckg.GetXaxis().SetTitle('Response')
        signalOverBckg.GetXaxis().SetLabelSize(0.15)
        signalOverBckg.GetXaxis().SetTitleSize(0.15)
        signalOverBckg.GetXaxis().SetTitleOffset(1.)
        signalOverBckg.SetFillStyle(0)
        signalOverBckg.SetMarkerStyle(2)
        signalOverBckg.SetMarkerColor(1)
        signalOverBckg.Draw('P')
        c1.Update()
        outfile_name = 'MCDNN_Response_Applied_notStacked_%s_%s.pdf'%(suffix,norm_suffix)
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
    #classifier_parent_dir = 'MultiClass_DNN_%sVars_%sHLs_%s_%s-VarTrans_%s-learnRate_%s-epochs' % (str(num_inputs),str(number_of_hidden_layers),activation_function,new_var_transform_name,str(learning_rate),num_epochs)
    classifier_parent_dir = 'MultiClass_DNN_%s' % (classifier_suffix)
    classifier_samples_dir = classifier_parent_dir+"/outputs"
    input_name = '%s/Applied_%s.root' % (classifier_samples_dir,classifier_parent_dir)
    print 'input_name: ', input_name
    input_file = TFile.Open(input_name)
    input_file.ls()

    histo_ttHCategory_ttHsample = input_file.Get('histo_ttHCategory_events_ttH')
    histo_ttHCategory_ttVsample = input_file.Get('histo_ttHCategory_events_ttV')
    histo_ttHCategory_ttJsample = input_file.Get('histo_ttHCategory_events_ttJets')

    histo_ttVCategory_ttHsample = input_file.Get('histo_ttVCategory_events_ttH')
    histo_ttVCategory_ttVsample = input_file.Get('histo_ttVCategory_events_ttV')
    histo_ttVCategory_ttJsample = input_file.Get('histo_ttVCategory_events_ttJets')

    histo_ttJCategory_ttHsample = input_file.Get('histo_ttJCategory_events_ttH')
    histo_ttJCategory_ttVsample = input_file.Get('histo_ttJCategory_events_ttV')
    histo_ttJCategory_ttJsample = input_file.Get('histo_ttJCategory_events_ttJets')

    # Rebin hitograms so at least 1 background event in each bin.
    rebinned_histograms_ttHCategory = rebinHistograms(histo_ttHCategory_ttHsample, histo_ttHCategory_ttVsample, histo_ttHCategory_ttJsample)
    rebinned_histograms_ttVCategory = rebinHistograms(histo_ttVCategory_ttVsample, histo_ttVCategory_ttHsample, histo_ttVCategory_ttJsample)
    rebinned_histograms_ttJCategory = rebinHistograms(histo_ttJCategory_ttJsample, histo_ttJCategory_ttHsample, histo_ttJCategory_ttVsample)

    # Get array of signal over background for each of the bins in the new histogram.
    signalOverBckg_ttHCategory = GetSignalOverBackground(rebinned_histograms_ttHCategory[0], rebinned_histograms_ttHCategory[1], rebinned_histograms_ttHCategory[2])
    signalOverBckg_ttVCategory = GetSignalOverBackground(rebinned_histograms_ttVCategory[0], rebinned_histograms_ttVCategory[1], rebinned_histograms_ttVCategory[2])
    signalOverBckg_ttJCategory = GetSignalOverBackground(rebinned_histograms_ttJCategory[0], rebinned_histograms_ttJCategory[1], rebinned_histograms_ttJCategory[2])

    # Create histograms. First argument passed to method should be the signal for the category you wish to plot.
    make_plot(rebinned_histograms_ttHCategory[0],rebinned_histograms_ttHCategory[1],rebinned_histograms_ttHCategory[2], 'ttH_category', True, classifier_parent_dir, signalOverBckg_ttHCategory, False)
    make_plot(rebinned_histograms_ttVCategory[0],rebinned_histograms_ttVCategory[1],rebinned_histograms_ttVCategory[2], 'ttV_category', True, classifier_parent_dir, signalOverBckg_ttVCategory, False)
    make_plot(rebinned_histograms_ttJCategory[0],rebinned_histograms_ttJCategory[1],rebinned_histograms_ttJCategory[2], 'ttJ_category', True, classifier_parent_dir, signalOverBckg_ttJCategory, False)

    make_plot(rebinned_histograms_ttHCategory[0],rebinned_histograms_ttHCategory[1],rebinned_histograms_ttHCategory[2], 'ttH_category', False, classifier_parent_dir, signalOverBckg_ttHCategory, False)
    make_plot(rebinned_histograms_ttVCategory[0],rebinned_histograms_ttVCategory[1],rebinned_histograms_ttVCategory[2], 'ttV_category', False, classifier_parent_dir, signalOverBckg_ttVCategory, False)
    make_plot(rebinned_histograms_ttJCategory[0],rebinned_histograms_ttJCategory[1],rebinned_histograms_ttJCategory[2], 'ttJ_category', False, classifier_parent_dir, signalOverBckg_ttJCategory, False)

    make_plot(rebinned_histograms_ttHCategory[0],rebinned_histograms_ttHCategory[1],rebinned_histograms_ttHCategory[2], 'ttH_category', True, classifier_parent_dir, signalOverBckg_ttHCategory, True)
    make_plot(rebinned_histograms_ttVCategory[0],rebinned_histograms_ttVCategory[1],rebinned_histograms_ttVCategory[2], 'ttV_category', True, classifier_parent_dir, signalOverBckg_ttVCategory, True)
    make_plot(rebinned_histograms_ttJCategory[0],rebinned_histograms_ttJCategory[1],rebinned_histograms_ttJCategory[2], 'ttJ_category', True, classifier_parent_dir, signalOverBckg_ttJCategory, True)

    make_plot(rebinned_histograms_ttHCategory[0],rebinned_histograms_ttHCategory[1],rebinned_histograms_ttHCategory[2], 'ttH_category', False, classifier_parent_dir, signalOverBckg_ttHCategory, True)
    make_plot(rebinned_histograms_ttVCategory[0],rebinned_histograms_ttVCategory[1],rebinned_histograms_ttVCategory[2], 'ttV_category', False, classifier_parent_dir, signalOverBckg_ttVCategory, True)
    make_plot(rebinned_histograms_ttJCategory[0],rebinned_histograms_ttJCategory[1],rebinned_histograms_ttJCategory[2], 'ttJ_category', False, classifier_parent_dir, signalOverBckg_ttJCategory, True)

main()
