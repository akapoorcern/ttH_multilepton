import ROOT, os, optparse
from ROOT import TMVA, TFile, TString, TLegend, THStack, TLatex, TH1D
from array import array
from subprocess import call
from os.path import isfile
from collections import OrderedDict

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


def GetDataOverMC(stack_mc, histo_data):

    #First on stack goes on bottom.
    DOverMC = histo_data.Clone('ratioframe')
    DOverMC.Divide(stack_mc.GetStack().Last())
    DOverMC.GetYaxis()
    DOverMC_maxY = DOverMC.GetMaximum()
    DOverMC_minY = DOverMC.GetMinimum()
    DOverMC.GetYaxis().SetTitle('Data/MC')
    DOverMC_maximum = DOverMC_maxY+(DOverMC_maxY*0.1)
    DOverMC_minimum = DOverMC_minY+(DOverMC_minY*0.1)
    if DOverMC_maximum < 2.:
        DOverMC.GetYaxis().SetRangeUser(0.5,DOverMC_maxY+(DOverMC_maxY*0.1))
    else:
        DOverMC.GetYaxis().SetRangeUser(0.5,2.)
    DOverMC.GetYaxis().SetNdivisions(6)
    DOverMC.GetYaxis().SetLabelSize(0.12)
    DOverMC.GetYaxis().SetTitleSize(0.12)
    DOverMC.GetYaxis().SetTitleOffset(0.2)
    DOverMC.GetXaxis().SetTitle('DNN Response')
    DOverMC.GetXaxis().SetLabelSize(0.15)
    DOverMC.GetXaxis().SetTitleSize(0.15)
    DOverMC.GetXaxis().SetTitleOffset(1.)
    DOverMC.SetFillStyle(0)
    DOverMC.SetMarkerStyle(2)
    DOverMC.SetMarkerColor(1)
    DOverMC.SetLineColor(1)
    return DOverMC


def rebinHistograms(hist_list, data_hist):

    nBins = hist_list.get('TTW').GetNbinsX()
    x_bin_edges = []
    #x_bin_edges.append(1.)
    x_bin_edges.append(0.)
    cumulative_sum_entries = 0

    #for x_bin_index in reversed(xrange(0,nBins)):
    for x_bin_index in xrange(0,nBins):
        tmp_bin_content_ttHww = hist_list.get('TTH_hww').GetBinContent(x_bin_index)
        tmp_bin_content_ttHtt = hist_list.get('TTH_htt').GetBinContent(x_bin_index)
        tmp_bin_content_ttHot = hist_list.get('TTH_hot').GetBinContent(x_bin_index)
        tmp_bin_content_ttHmm = hist_list.get('TTH_hmm').GetBinContent(x_bin_index)
        tmp_bin_content_conv = hist_list.get('Conv').GetBinContent(x_bin_index)
        for hist_key, hist in hist_list.iteritems():
            if 'TTH' in hist_key or 'Data' in hist_key: continue
            cumulative_sum_entries = cumulative_sum_entries + hist.GetBinContent(x_bin_index)
        if tmp_bin_content_ttHww >= 0 and tmp_bin_content_ttHtt >= 0 and tmp_bin_content_ttHot >= 0 and tmp_bin_content_ttHmm >= 0 and cumulative_sum_entries>=0:
        #if tmp_bin_content_ttHww + tmp_bin_content_ttHtt + tmp_bin_content_ttHot + tmp_bin_content_ttHmm >= 0 and cumulative_sum_entries >= 0:
            print 'cumulative_sum_entries = ', cumulative_sum_entries
            #new_x_bin_edge = hist_list.get('TTW').GetXaxis().GetBinLowEdge(x_bin_index)
            new_x_bin_edge = hist_list.get('TTW').GetXaxis().GetBinUpEdge(x_bin_index)
            x_bin_edges.append(new_x_bin_edge)
            cumulative_sum_entries = 0
        #if tmp_bin_content_ttHww >= 0 and tmp_bin_content_ttHtt >= 0 and tmp_bin_content_ttHot >= 0 and tmp_bin_content_ttHmm >= 0 and x_bin_index == 0 and cumulative_sum_entries>=0:
        #    x_bin_edges = x_bin_edges[:-1]
        #    x_bin_edges.append(0.)

    x_bin_edges.append(1.)

    #if x_bin_edges[len(x_bin_edges)-1] != 0:
    #    x_bin_edges.append(0.)
    ordered_bin_edges = []

    #for x in reversed(x_bin_edges):
    '''for x in x_bin_edges:
        ordered_bin_edges.append(x)'''

    ordered_bin_edges = [0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.6125,0.6875,0.75,0.8125,0.875,0.9375,1.0]

    n_xbins = len(ordered_bin_edges)-1
    x_bin_edges_fuckingroot = array('d',ordered_bin_edges)

    new_hists = OrderedDict()
    for hist_key, hist in hist_list.iteritems():
        new_hists[hist_key] = hist.Rebin(n_xbins,hist.GetName(),x_bin_edges_fuckingroot)
        print 'Process hist: ' new_hists[hist_key].GetName()
        for x_bin_index in xrange(0,new_hists[hist_key].GetNbinsX()):
            tmp_bin_content = new_hists[hist_key].GetBinContent(x_bin_index)
            print 'bin # %i , value %f' % (x_bin_index,tmp_bin_content)
    new_hists['Data'] = data_hist.Rebin(n_xbins,data_hist.GetName(),x_bin_edges_fuckingroot)

    hist_list.clear()
    return new_hists

# Makes plots of the various categories for the DNN. Each 'category' histogram is a plot of the response on one of the DNN nodes.
# However, it contains only the events that showed the highest DNN response for said node.
def make_plot(stacked_hist, suffix, classifier_parent_dir, norm, legend, data_hist = None, ratioplot = None):

    c1 = ROOT.TCanvas('c1',',1000,1000')
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
    norm_suffix = 'LumiNorm'

    classifier_plots_dir = classifier_parent_dir+"/plots-2018-11-15"
    print 'using plots dir: ', classifier_plots_dir
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    stacked_hist.Draw("HIST")
    stacked_hist.GetYaxis().SetTitle('Events/Bin')
    stacked_hist.GetXaxis().SetTitle('DNN Output')

    if data_hist != None:
        data_hist.SetMarkerStyle(20)
        data_hist.SetMarkerColor(1)
        data_hist.SetLineColor(1)
        data_hist.Draw("HISTEPSAME")

    legend.Draw("same")

    txt2=ROOT.TLatex()
    txt2.SetNDC(True)
    txt2.SetTextFont(43)
    txt2.SetTextSize(18)
    txt2.SetTextAlign(12)
    txt2.DrawLatex(0.13,0.925,'#bf{CMS}')
    txt2.DrawLatex(0.2,0.92,'#it{Preliminary}')
    txt2.DrawLatex(0.57,0.925,'%3.1f fb^{-1} (13 TeV)' %(41860.080/1000.) )

    # Add custom title
    l1=ROOT.TLatex()
    l1.SetNDC();
    l1.SetTextFont(43)
    l1.SetTextSize(15)
    l1.SetTextAlign(12)
    latex_title = ''
    if 'em_ttHCategory' in suffix:
        latex_title = "#it{prefit: em ttH Category}"
    elif 'em_ttVCategory' in suffix:
        latex_title = "#it{prefit: em ttV Category}"
    elif 'em_ttJCategory' in suffix:
        latex_title = "#it{prefit: em ttJ Category}"
    elif 'mm_ttHCategory' in suffix:
        latex_title = "#it{prefit: mm ttH Category}"
    elif 'mm_ttVCategory' in suffix:
        latex_title = "#it{prefit: mm ttV Category}"
    elif 'mm_ttJCategory' in suffix:
        latex_title = "#it{prefit: mm ttJ Category}"
    else:
        latex_title = "#it{prefit: ee Category}"

    l1.DrawLatex(0.15,0.8,latex_title)

    # Draw Data/MC ratio plot
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

    if ratioplot != None:

        ratioplot_maxY = ratioplot.GetMaximum()
        ratioplot.GetYaxis().SetTitle('Data/MC')
        ratioplot_maximum = ratioplot_maxY+(ratioplot_maxY*0.1)
        if ratioplot_maximum < 2.:
            ratioplot.GetYaxis().SetRangeUser(0.,ratioplot_maxY+(ratioplot_maxY*0.1))
        else:
            ratioplot.GetYaxis().SetRangeUser(0.,3.)
        ratioplot.GetYaxis().SetNdivisions(6)
        ratioplot.GetYaxis().SetLabelSize(0.12)
        ratioplot.GetYaxis().SetTitleSize(0.12)
        ratioplot.GetYaxis().SetTitleOffset(0.2)
        ratioplot.GetXaxis().SetTitle('Response')
        ratioplot.GetXaxis().SetLabelSize(0.15)
        ratioplot.GetXaxis().SetTitleSize(0.15)
        ratioplot.GetXaxis().SetTitleOffset(1.)
        ratioplot.SetFillStyle(0)
        ratioplot.SetMarkerStyle(2)
        ratioplot.SetMarkerColor(1)
        ratioplot.SetLineColor(1)
        ratioplot.Draw("P")
        line = ROOT.TLine(0,1,1,1);
        line.SetLineColor(2);
        line.Draw("same");
        c1.Update()

    outfile_name = 'MCDNN_Response_Evaluated_stacked_%s_%s.pdf'%(suffix,norm_suffix)
    output_fullpath = classifier_plots_dir + '/' + outfile_name
    c1.Print(output_fullpath,'pdf')

def main():
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify which network training directory to get inputs from',      default=None,        type='string')
    parser.add_option('-d', '--data',        dest='data_flag'  ,      help='1 = include data from plots, 0 = exclude data from plots',      default=0,        type='int')
    (opt, args) = parser.parse_args()

    if opt.input_suffix == None:
        print 'Input files suffix not defined!'
        sys.exit(1)

    data_flag = opt.data_flag

    classifier_suffix = opt.input_suffix
    classifier_parent_dir = 'V7-DNN_%s' % (classifier_suffix)
    classifier_samples_dir = classifier_parent_dir+"/outputs-newbinning"
    print 'Reading samples from: ', classifier_samples_dir

    sample_name = ['Conv','EWK','Fakes','Flips','Rares','TTH_hww','TTH_htt','TTH_hot','TTW','TTZ','TTWW','TTH_hmm','Data']
    channel_name = ['2L']

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
    input_name_TTH_hww = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[5],channel_name[0])
    input_file_TTH_hww = TFile.Open(input_name_TTH_hww)
    input_name_TTH_htt = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[6],channel_name[0])
    input_file_TTH_htt = TFile.Open(input_name_TTH_htt)
    input_name_TTH_hot = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[7],channel_name[0])
    input_file_TTH_hot = TFile.Open(input_name_TTH_hot)
    input_name_TTW = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[8],channel_name[0])
    input_file_TTW = TFile.Open(input_name_TTW)
    input_name_TTZ = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[9],channel_name[0])
    input_file_TTZ = TFile.Open(input_name_TTZ)
    input_name_TTWW = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[10],channel_name[0])
    input_file_TTWW = TFile.Open(input_name_TTWW)
    input_name_TTH_hmm = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[11],channel_name[0])
    input_file_TTH_hmm = TFile.Open(input_name_TTH_hmm)
    input_name_data = '%s/%s/Evaluated_%s_%s_%s.root' % (classifier_samples_dir,channel_name[0],classifier_suffix,sample_name[12],channel_name[0])
    input_file_data = TFile.Open(input_name_data)

    categories = ['ee','em_ttHCategory','em_ttVCategory','em_ttJCategory','mm_ttHCategory','mm_ttVCategory','mm_ttJCategory']
    #categories = ['mm_ttHCategory']

    for cat in categories:
        print 'category: ' , cat

        process_list = {
        "Conv" : 5,
        "EWK" : 6,
        "Fakes" : 13,
        "Flips" : 17,
        "Rares" : 38,
        "TTW" : 8,
        "TTZ" : 9,
        "TTWW" : 3,
        "TTH_hww" : 2,
        "TTH_htt" : 2,
        "TTH_hot" : 2,
        "TTH_hmm" : 2,
        "Data":1
        }

        histo_Conv_name = 'histo_%s_events_Conv_%s' % (cat,channel_name[0])
        histo_EWK_name = 'histo_%s_events_EWK_%s' % (cat,channel_name[0])
        histo_Fakes_name = 'histo_%s_events_Fakes_%s' % (cat,channel_name[0])
        histo_Flips_name = 'histo_%s_events_Flips_%s' % (cat,channel_name[0])
        histo_Rares_name = 'histo_%s_events_Rares_%s' % (cat,channel_name[0])
        histo_TTH_hww_name = 'histo_%s_events_TTH_hww_%s' % (cat,channel_name[0])
        histo_TTH_htt_name = 'histo_%s_events_TTH_htt_%s' % (cat,channel_name[0])
        histo_TTH_hot_name = 'histo_%s_events_TTH_hot_%s' % (cat,channel_name[0])
        histo_TTH_hmm_name = 'histo_%s_events_TTH_hmm_%s' % (cat,channel_name[0])
        histo_TTW_name = 'histo_%s_events_TTW_%s' % (cat,channel_name[0])
        histo_TTZ_name = 'histo_%s_events_TTZ_%s' % (cat,channel_name[0])
        histo_TTWW_name = 'histo_%s_events_TTWW_%s' % (cat,channel_name[0])
        histo_data_name = 'histo_%s_events_Data' % (cat)

        histo_Conv_sample = input_file_Conv.Get(histo_Conv_name)
        histo_EWK_sample = input_file_EWK.Get(histo_EWK_name)
        histo_Fakes_sample = input_file_Fakes.Get(histo_Fakes_name)
        histo_Flips_sample = input_file_Flips.Get(histo_Flips_name)
        histo_Rares_sample = input_file_Rares.Get(histo_Rares_name)
        histo_TTH_hww_sample = input_file_TTH_hww.Get(histo_TTH_hww_name)
        histo_TTH_htt_sample = input_file_TTH_htt.Get(histo_TTH_htt_name)
        histo_TTH_hot_sample = input_file_TTH_hot.Get(histo_TTH_hot_name)
        histo_TTH_hmm_sample = input_file_TTH_hmm.Get(histo_TTH_hmm_name)
        histo_TTW_sample = input_file_TTW.Get(histo_TTW_name)
        histo_TTZ_sample = input_file_TTZ.Get(histo_TTZ_name)
        histo_TTWW_sample = input_file_TTWW.Get(histo_TTWW_name)
        histo_data_sample = input_file_data.Get(histo_data_name)

        # Turn this into dictionary
        hist_list = OrderedDict([
        ("Conv" , histo_Conv_sample),
        ("EWK" , histo_EWK_sample),
        ("Fakes" , histo_Fakes_sample),
        ("Flips" , histo_Flips_sample),
        ("Rares" , histo_Rares_sample),
        ("TTW" , histo_TTW_sample),
        ("TTZ" , histo_TTZ_sample),
        ("TTWW" , histo_TTWW_sample),
        ("TTH_hww" , histo_TTH_hww_sample),
        ("TTH_htt" , histo_TTH_htt_sample),
        ("TTH_hot" , histo_TTH_hot_sample),
        ("TTH_hmm" , histo_TTH_hmm_sample)
        ])

        # Rebin Histograms so > 0 total background entries per bin.
        # Returns array of histograms in same order as was passed to the function.
        rebinned_histograms = rebinHistograms(hist_list, histo_data_sample)

        hist_stack = ROOT.THStack()
        legend = TLegend(0.7,  0.7,  0.9,  0.9)
        legend.SetNColumns(2)
        for rebinned_hist_name, rebinned_hist in rebinned_histograms.iteritems():
            print 'rebinned_hist_name = ', rebinned_hist_name
            rebinned_hist.SetMarkerColor(process_list[rebinned_hist_name])
            rebinned_hist.SetLineColor(process_list[rebinned_hist_name])
            rebinned_hist.GetYaxis().SetTitle("Events/bin")
            rebinned_hist.SetMarkerStyle(20)
            rebinned_hist.Sumw2()
            if 'Data' in rebinned_hist_name:
                legend.AddEntry(rebinned_hist,rebinned_hist_name,'p')
                continue
            else:
                legend.AddEntry(rebinned_hist,rebinned_hist_name,'f')
                rebinned_hist.SetMarkerStyle(20)
                if 'TTH_hww' in rebinned_hist_name:
                    rebinned_hist.SetFillStyle(2001)
                if 'TTH_hmm' in rebinned_hist_name:
                    rebinned_hist.SetFillStyle(1001)
                if 'TTH_htt' in rebinned_hist_name:
                    rebinned_hist.SetFillStyle(4001)
                if 'TTH_hot' in rebinned_hist_name:
                    rebinned_hist.SetFillStyle(5001)
                else:
                    rebinned_hist.SetFillStyle(3001)
                rebinned_hist.SetFillColor(process_list[rebinned_hist_name])
                print 'Adding hist'
                hist_stack.Add(rebinned_hist)

        hist_stack.SetMaximum(hist_stack.GetStack().Last().GetMaximum() + (hist_stack.GetStack().Last().GetMaximum()/2))
        hist_stack.SetMinimum(0.)

        # Get Data/MC agreement
        # Ensure to pass data histogram as last argument.
        if data_flag:
            dataOverMC = GetDataOverMC(hist_stack,rebinned_histograms['Data'])
            # Create histograms.
            make_plot(hist_stack, cat, classifier_parent_dir, False, legend, rebinned_histograms['Data'] , dataOverMC)
        else:
            make_plot(hist_stack, cat, classifier_parent_dir, False, legend)

main()
