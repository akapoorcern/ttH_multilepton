import ROOT, os, optparse
from ROOT import TMVA, TFile, TString, TLegend, THStack, TLatex, TH1D
from array import array
from subprocess import call
from os.path import isfile

def GetHistoScale(histo):
    dx_hist_train = (histo.GetXaxis().GetXmax() - histo.GetXaxis().GetXmin()) / histo.GetNbinsX()
    hist_train_norm = 1./histo.GetSumOfWeights()/dx_hist_train
    histo.Scale(hist_train_norm)
    return histo

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
#def rebinHistograms(original_hist_sig, original_hist_bckg1, original_hist_bckg2, original_hist_bckg3, original_hist_bckg4, original_hist_bckg5, original_hist_bckg6, original_hist_bckg7, original_hist_bckg8,original_hist_bckg9):
def rebinHistograms(original_hist_sig, original_hist_bckg1, original_hist_bckg2, original_hist_bckg3, original_hist_bckg4, original_hist_bckg5, original_hist_bckg6, original_hist_bckg7, original_hist_bckg8):
    nBins = original_hist_sig.GetNbinsX()
    x_bin_edges = []
    x_bin_edges.append(0.)
    x_axis_bckg1 = original_hist_bckg1.GetXaxis()
    cumulative_sum_entries = 0

    for x_bin_index in xrange(1,nBins):
        #nentries_sig = original_hist_sig.GetBinContent(x_bin_index)
        nentries_bckg1 = original_hist_bckg1.GetBinContent(x_bin_index)
        nentries_bckg2 = original_hist_bckg2.GetBinContent(x_bin_index)
        nentries_bckg3 = original_hist_bckg3.GetBinContent(x_bin_index)
        nentries_bckg4 = original_hist_bckg4.GetBinContent(x_bin_index)
        nentries_bckg5 = original_hist_bckg5.GetBinContent(x_bin_index)
        nentries_bckg6 = original_hist_bckg6.GetBinContent(x_bin_index)
        nentries_bckg7 = original_hist_bckg7.GetBinContent(x_bin_index)

        cumulative_sum_entries = cumulative_sum_entries + nentries_bckg1 + nentries_bckg2 + nentries_bckg3 + nentries_bckg4 + nentries_bckg5 + nentries_bckg6 + nentries_bckg7
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
    HistBckg3_rebin = original_hist_bckg3.Rebin(n_xbins,'HistBckg3_rebin',x_bin_edges_fuckingroot)
    HistBckg4_rebin = original_hist_bckg4.Rebin(n_xbins,'HistBckg4_rebin',x_bin_edges_fuckingroot)
    HistBckg5_rebin = original_hist_bckg5.Rebin(n_xbins,'HistBckg5_rebin',x_bin_edges_fuckingroot)
    HistBckg6_rebin = original_hist_bckg6.Rebin(n_xbins,'HistBckg6_rebin',x_bin_edges_fuckingroot)
    HistBckg7_rebin = original_hist_bckg7.Rebin(n_xbins,'HistBckg7_rebin',x_bin_edges_fuckingroot)
    HistBckg8_rebin = original_hist_bckg8.Rebin(n_xbins,'HistBckg8_rebin',x_bin_edges_fuckingroot)
    #HistBckg9_rebin = original_hist_bckg9.Rebin(n_xbins,'HistBckg9_rebin',x_bin_edges_fuckingroot)

    new_hists = [HistSig_rebin, HistBckg1_rebin, HistBckg2_rebin, HistBckg3_rebin, HistBckg4_rebin, HistBckg5_rebin, HistBckg6_rebin, HistBckg7_rebin, HistBckg8_rebin]
    return new_hists

# Makes plots of the various categories for the DNN. Each 'category' histogram is a plot of the response on one of the DNN nodes.
# However, it contains only the events that showed the highest DNN response for said node.
def make_plot(histo_0, histo_1, histo_2, histo_3, histo_4, histo_5, histo_6, histo_7, histo_8, suffix, stacked, classifier_parent_dir, signalOverBckg, norm):

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

    signal_colour = 0
    bckg1_colour = 0
    bckg2_colour = 0
    bckg3_colour = 0
    bckg4_colour = 0
    bckg5_colour = 0
    bckg6_colour = 0
    bckg7_colour = 0
    bckg8_colour = 0
    legend = TLegend(0.85,  0.85,  1.,  1.)
    if suffix == 'ttHCategory':
        signal_colour = 2
        bckg1_colour = 8
        bckg2_colour = 3
        bckg3_colour = 5
        bckg4_colour = 6
        bckg5_colour = 13
        bckg6_colour = 17
        bckg7_colour = 38
        #bckg8_colour = 30
        bckg8_colour = 1
        legend.AddEntry(histo_0,"ttH sample")
        legend.AddEntry(histo_1,"ttW sample")
        legend.AddEntry(histo_2,"ttZ sample")
        legend.AddEntry(histo_3,"Conv. sample")
        legend.AddEntry(histo_4,"EWK sample")
        legend.AddEntry(histo_5,"Fake sample")
        legend.AddEntry(histo_6,"Flips sample")
        legend.AddEntry(histo_7,"Rares sample")
        #legend.AddEntry(histo_8,"ttWW sample")
        legend.AddEntry(histo_8,"data sample")
        # Add custom title
        l1=ROOT.TLatex()
        l1.SetNDC();
        latex_title = "ttH Category"
    if suffix == 'ttVCategory':
        signal_colour = 2
        bckg1_colour = 8
        bckg2_colour = 3
        bckg3_colour = 5
        bckg4_colour = 6
        bckg5_colour = 13
        bckg6_colour = 17
        bckg7_colour = 38
        #bckg8_colour = 30
        bckg8_colour = 1
        legend.AddEntry(histo_0,"ttH sample")
        legend.AddEntry(histo_1,"ttW sample")
        legend.AddEntry(histo_2,"ttZ sample")
        legend.AddEntry(histo_3,"Conv. sample")
        legend.AddEntry(histo_4,"EWK sample")
        legend.AddEntry(histo_5,"Fake sample")
        legend.AddEntry(histo_6,"Flips sample")
        legend.AddEntry(histo_7,"Rares sample")
        #legend.AddEntry(histo_8,"ttWW sample")
        legend.AddEntry(histo_8,"data sample")
        # Add custom title
        l1=ROOT.TLatex()
        l1.SetNDC();
        latex_title = "ttV Category"
    if suffix == 'ttJCategory':
        signal_colour = 2
        bckg1_colour = 8
        bckg2_colour = 3
        bckg3_colour = 5
        bckg4_colour = 6
        bckg5_colour = 13
        bckg6_colour = 17
        bckg7_colour = 38
        #bckg8_colour = 30
        bckg8_colour = 1
        legend.AddEntry(histo_0,"ttH sample")
        legend.AddEntry(histo_1,"ttW sample")
        legend.AddEntry(histo_2,"ttZ sample")
        legend.AddEntry(histo_3,"Conv. sample")
        legend.AddEntry(histo_4,"EWK sample")
        legend.AddEntry(histo_5,"Fake sample")
        legend.AddEntry(histo_6,"Flips sample")
        legend.AddEntry(histo_7,"Rares sample")
        #legend.AddEntry(histo_8,"ttWW sample")
        legend.AddEntry(histo_8,"data sample")
        # Add custom title
        l1=ROOT.TLatex()
        l1.SetNDC();
        latex_title = "tt+Jets Category"

    norm_suffix = ''
    if norm == True:
        histo_0.GetYaxis().SetTitle("(1/N)dN/dX")
        histo_1.GetYaxis().SetTitle("(1/N)dN/dX")
        histo_2.GetYaxis().SetTitle("(1/N)dN/dX")
        histo_3.GetYaxis().SetTitle("(1/N)dN/dX")
        histo_4.GetYaxis().SetTitle("(1/N)dN/dX")
        histo_5.GetYaxis().SetTitle("(1/N)dN/dX")
        histo_6.GetYaxis().SetTitle("(1/N)dN/dX")
        histo_7.GetYaxis().SetTitle("(1/N)dN/dX")
        histo_8.GetYaxis().SetTitle("(1/N)dN/dX")
        #histo_9.GetYaxis().SetTitle("(1/N)dN/dX")
        norm_suffix = 'Norm'

        histo_0 = GetHistoScale(histo_0)
        histo_1 = GetHistoScale(histo_1)
        histo_2 = GetHistoScale(histo_2)
        histo_3 = GetHistoScale(histo_3)
        histo_4 = GetHistoScale(histo_4)
        histo_5 = GetHistoScale(histo_5)
        histo_6 = GetHistoScale(histo_6)
        histo_7 = GetHistoScale(histo_7)
        histo_8 = GetHistoScale(histo_8)
        #histo_9 = GetHistoScale(histo_9)
    else:
        histo_0.GetYaxis().SetTitle("Events/bin")
        norm_suffix = 'LumiNorm'

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

    histo_3.SetLineColor(bckg3_colour)
    histo_3.SetMarkerColor(bckg3_colour)
    histo_3.SetMarkerStyle(20)
    histo_3.SetFillColor(bckg3_colour)
    histo_3.SetFillStyle(3001)

    histo_4.SetLineColor(bckg4_colour)
    histo_4.SetMarkerColor(bckg4_colour)
    histo_4.SetMarkerStyle(20)
    histo_4.SetFillColor(bckg4_colour)
    histo_4.SetFillStyle(3001)

    histo_5.SetLineColor(bckg5_colour)
    histo_5.SetMarkerColor(bckg5_colour)
    histo_5.SetMarkerStyle(20)
    histo_5.SetFillColor(bckg5_colour)
    histo_5.SetFillStyle(3001)

    histo_6.SetLineColor(bckg6_colour)
    histo_6.SetMarkerColor(bckg6_colour)
    histo_6.SetMarkerStyle(20)
    histo_6.SetFillColor(bckg6_colour)
    histo_6.SetFillStyle(3001)

    histo_7.SetLineColor(bckg7_colour)
    histo_7.SetMarkerColor(bckg7_colour)
    histo_7.SetMarkerStyle(20)
    histo_7.SetFillColor(bckg7_colour)
    histo_7.SetFillStyle(3001)

    '''histo_8.SetLineColor(bckg8_colour)
    histo_8.SetMarkerColor(bckg8_colour)
    histo_8.SetMarkerStyle(20)
    histo_8.SetFillColor(bckg8_colour)
    histo_8.SetFillStyle(3001)'''

    histo_8.SetLineColor(bckg8_colour)
    histo_8.SetMarkerColor(bckg8_colour)
    histo_8.SetMarkerStyle(20)

    classifier_plots_dir = classifier_parent_dir+"/plots"
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    if stacked == True:
        hs = ROOT.THStack()
        #First on stack goes on bottom.
        #hs.Add(histo_8)
        hs.Add(histo_7)
        hs.Add(histo_6)
        hs.Add(histo_5)
        hs.Add(histo_4)
        hs.Add(histo_3)
        hs.Add(histo_2)
        hs.Add(histo_1)
        hs.Add(histo_0)

        maxyaxis = max(histo_0.GetMaximum(), histo_1.GetMaximum(), histo_2.GetMaximum(), histo_3.GetMaximum(), histo_4.GetMaximum(), histo_5.GetMaximum(), histo_6.GetMaximum(), histo_7.GetMaximum(), histo_8.GetMaximum())
        #maxyaxis = max(histo_0.GetMaximum(), histo_1.GetMaximum(), histo_2.GetMaximum(), histo_3.GetMaximum(), histo_4.GetMaximum(), histo_5.GetMaximum(), histo_6.GetMaximum(), histo_7.GetMaximum(), histo_8.GetMaximum(), histo_9.GetMaximum())
        hs.SetMaximum(maxyaxis + 5.)

        hs.Draw("HIST")
        #histo_9.Draw("HISTEPSAME")
        histo_8.Draw("HISTEPSAME")

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
        outfile_name = 'MCDNN_Response_Evaluated_stacked_%s_%s.pdf'%(suffix,norm_suffix)
        output_fullpath = classifier_plots_dir + '/' + outfile_name
        c1.Print(output_fullpath,'pdf')

    elif stacked == False:
        if suffix == 'ttH_category':
            histo_0.SetFillStyle(0)
            histo_1.SetFillStyle(0)
            histo_2.SetFillStyle(0)
            histo_3.SetFillStyle(0)
            histo_4.SetFillStyle(0)
            histo_5.SetFillStyle(0)
            histo_6.SetFillStyle(0)
            histo_7.SetFillStyle(0)
            #histo_8.SetFillStyle(0)
            histo_1.Draw("HIST")
            histo_2.Draw("HISTSAME")
            histo_3.Draw("HISTSAME")
            histo_4.Draw("HISTSAME")
            histo_5.Draw("HISTSAME")
            histo_6.Draw("HISTSAME")
            histo_7.Draw("HISTSAME")
            histo_8.Draw("HISTSAMEP")
            #histo_9.Draw("HISTSAMEP")
        if suffix == 'ttV_category':
            histo_0.SetFillStyle(0)
            histo_1.SetFillStyle(0)
            histo_2.SetFillStyle(0)
            histo_3.SetFillStyle(0)
            histo_4.SetFillStyle(0)
            histo_5.SetFillStyle(0)
            histo_6.SetFillStyle(0)
            histo_7.SetFillStyle(0)
            #histo_8.SetFillStyle(0)
            histo_0.Draw("HIST")
            histo_2.Draw("HISTSAME")
            histo_1.Draw("HISTSAME")
            histo_3.Draw("HISTSAME")
            histo_4.Draw("HISTSAME")
            histo_5.Draw("HISTSAME")
            histo_6.Draw("HISTSAME")
            histo_7.Draw("HISTSAME")
            histo_8.Draw("HISTSAMEP")
            #histo_9.Draw("HISTSAMEP")
        if suffix == 'ttJ_category':
            histo_0.SetFillStyle(0)
            histo_1.SetFillStyle(0)
            histo_2.SetFillStyle(0)
            histo_3.SetFillStyle(0)
            histo_4.SetFillStyle(0)
            histo_5.SetFillStyle(0)
            histo_6.SetFillStyle(0)
            histo_7.SetFillStyle(0)
            #histo_8.SetFillStyle(0)

            histo_0.Draw("HIST")
            histo_2.Draw("HISTSAME")
            histo_1.Draw("HISTSAME")
            histo_3.Draw("HISTSAME")
            histo_4.Draw("HISTSAME")
            histo_5.Draw("HISTSAME")
            histo_6.Draw("HISTSAME")
            histo_7.Draw("HISTSAME")
            histo_8.Draw("HISTSAMEP")
            #histo_9.Draw("HISTSAMEP")

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
        outfile_name = 'MCDNN_Response_Evaluated_notStacked_%s_%s.pdf'%(suffix,norm_suffix)
        output_fullpath = classifier_plots_dir + '/' + outfile_name
        c1.Print(output_fullpath,'pdf')

def main():
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify which network training directory to get inputs from',      default=None,        type='string')
    (opt, args) = parser.parse_args()

    if opt.input_suffix == None:
        print 'Input files suffix not defined!'
        sys.exit(1)

    classifier_suffix = opt.input_suffix
    classifier_parent_dir = 'MultiClass_DNN_17Vars_%s' % (classifier_suffix)
    classifier_samples_dir = classifier_parent_dir+"/outputs"
    print 'Reading samples from: ', classifier_samples_dir

    sample_name = ['Conv','EWK','Fakes','Flips','Rares','TTH','TTW','TTZ','data']
    channel_name = ['2LSS']
    #channel_name = ['ttWctrl']

    input_name_Conv = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[0],channel_name[0])
    input_file_Conv = TFile.Open(input_name_Conv)
    input_name_EWK = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[1],channel_name[0])
    input_file_EWK = TFile.Open(input_name_EWK)
    input_name_Fakes = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[2],channel_name[0])
    input_file_Fakes = TFile.Open(input_name_Fakes)
    input_name_Flips = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[3],channel_name[0])
    input_file_Flips = TFile.Open(input_name_Flips)
    input_name_Rares = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[4],channel_name[0])
    input_file_Rares = TFile.Open(input_name_Rares)
    input_name_TTH = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[5],channel_name[0])
    input_file_TTH = TFile.Open(input_name_TTH)
    input_name_TTW = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[6],channel_name[0])
    input_file_TTW = TFile.Open(input_name_TTW)
    input_name_TTZ = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[7],channel_name[0])
    input_file_TTZ = TFile.Open(input_name_TTZ)
    input_name_data = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[8],channel_name[0])
    input_file_data = TFile.Open(input_name_data)
    #input_name_TTWW = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[9],channel_name[0])
    #input_file_TTWW = TFile.Open(input_name_TTWW)

    #histo_name = 'histo_ttHCategory_events_%s_%s' % (sample_name[0],channel_name)

    categories = ['ttHCategory','ttVCategory','ttJCategory']
    print 'input_name_TTH: ', input_name_TTH
    for cat in categories:
        histo_Conv_name = 'histo_%s_events_Conv_%s' % (cat,channel_name[0])
        histo_EWK_name = 'histo_%s_events_EWK_%s' % (cat,channel_name[0])
        histo_Fakes_name = 'histo_%s_events_Fakes%s' % (cat,channel_name[0])
        histo_Flips_name = 'histo_%s_events_Flips_%s' % (cat,channel_name[0])
        histo_Rares_name = 'histo_%s_events_Rares_%s' % (cat,channel_name[0])
        histo_TTH_name = 'histo_%s_events_TTH_%s' % (cat,channel_name[0])
        histo_TTW_name = 'histo_%s_events_TTW_%s' % (cat,channel_name[0])
        histo_TTZ_name = 'histo_%s_events_TTZ_%s' % (cat,channel_name[0])
        histo_data_name = 'histo_%s_events_' % (cat)
        print 'histo_data_name: ', histo_data_name
        #histo_TTWW_name = 'histo_%s_events_TTWW_%s' % (cat,channel_name[0])

        histo_Conv_sample = input_file_Conv.Get(histo_Conv_name)
        histo_EWK_sample = input_file_EWK.Get(histo_EWK_name)
        histo_Fakes_sample = input_file_Fakes.Get(histo_Fakes_name)
        histo_Flips_sample = input_file_Flips.Get(histo_Flips_name)
        histo_Rares_sample = input_file_Rares.Get(histo_Rares_name)
        histo_TTH_sample = input_file_TTH.Get(histo_TTH_name)
        histo_TTW_sample = input_file_TTW.Get(histo_TTW_name)
        histo_TTZ_sample = input_file_TTZ.Get(histo_TTZ_name)
        histo_data_sample = input_file_data.Get(histo_data_name)
        histo_Conv_sample.Sumw2()
        histo_EWK_sample.Sumw2()
        histo_Fakes_sample.Sumw2()
        histo_Flips_sample.Sumw2()
        histo_Rares_sample.Sumw2()
        histo_TTH_sample.Sumw2()
        histo_TTW_sample.Sumw2()
        histo_TTZ_sample.Sumw2()
        histo_data_sample.Sumw2()

        #rebinned_histograms = rebinHistograms(histo_TTH_sample, histo_TTW_sample, histo_TTZ_sample, histo_Conv_sample, histo_EWK_sample, histo_Fakes_sample, histo_Flips_sample, histo_Rares_sample, histo_data_sample, histo_TTWW_sample)
        rebinned_histograms = rebinHistograms(histo_TTH_sample, histo_TTW_sample, histo_TTZ_sample, histo_Conv_sample, histo_EWK_sample, histo_Fakes_sample, histo_Flips_sample, histo_Rares_sample, histo_data_sample)

        # Get array of signal over background for each of the bins in the new histogram.
        signalOverBckg = GetSignalOverBackground(rebinned_histograms[0], rebinned_histograms[1], rebinned_histograms[2])

        # Create histograms. First argument passed to method should be the signal for the category you wish to plot.
        make_plot(rebinned_histograms[0],rebinned_histograms[1], rebinned_histograms[2], rebinned_histograms[3], rebinned_histograms[4], rebinned_histograms[5], rebinned_histograms[6], rebinned_histograms[7], rebinned_histograms[8], cat, True, classifier_parent_dir, signalOverBckg, False)

main()
