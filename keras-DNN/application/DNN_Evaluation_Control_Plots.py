import ROOT, os, argparse,math
import numpy as np
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

    # Histogram scaled per entry per unit of X
    nBins = hist_sig.GetNbinsX()
    #dX = (hist_sig.GetXaxis().GetXmax() - hist_sig.GetXaxis().GetXmin()) / hist_sig.GetNbinsX()
    nS = hist_sig.GetSumOfWeights()
    nB = hist_bckg.GetSumOfWeights()

    if nS == 0:
        print 'WARNING: no signal weights'
    if nB == 0:
        print 'WARNING: no bckg weights'
    sig_bin_norm_sum=0
    bckg_bin_norm_sum=0
    for i in xrange(1,nBins):
        if nS != 0:
            sig_bin_norm = hist_sig.GetBinContent(i)/nS
            sig_bin_norm_sum += sig_bin_norm
        else: continue
        if nB != 0 :
            bckg_bin_norm = hist_bckg.GetBinContent(i)/nB
            bckg_bin_norm_sum += bckg_bin_norm
        else: continue
        # Separation:
        if(sig_bin_norm+bckg_bin_norm > 0):
            separation += 0.5 * ((sig_bin_norm - bckg_bin_norm) * (sig_bin_norm - bckg_bin_norm)) / (sig_bin_norm + bckg_bin_norm)
    #separation *= dX

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

#def rebinHistograms(hist_list, data_hist):
def rebinHistograms(hist_list):
    print 'hist_list.get(ttW): ', hist_list.get('ttW')

    nBins = hist_list.get('ttW').GetNbinsX()

    x_bin_edges = []
    cumulative_bckg_entries = 0
    cumulative_sig_entries = 0
    combined_bckg_hist_values = []
    combined_sig_hist_values = []
    combined_bckg_hist = ROOT.TH1F('bckg_hists','bckg_hists',nBins,0,1)
    combined_sig_hist = ROOT.TH1F('sig_hists','sig_hists',nBins,0,1)
    for x_bin_index in xrange(0,nBins):
        tmp_bin_content_ttHww = hist_list.get('ttH_HWW').GetBinContent(x_bin_index)

        for hist_key, hist in hist_list.iteritems():
            print 'hist_key: ',hist_key
            print 'hist: ',hist
            if 'Data' in hist_key: continue
            elif 'ttH_' in hist_key:
                cumulative_sig_entries = cumulative_sig_entries + hist.GetBinContent(x_bin_index)
                combined_sig_hist.AddBinContent(x_bin_index,hist.GetBinContent(x_bin_index))
                continue
            else:
                cumulative_bckg_entries = cumulative_bckg_entries + hist.GetBinContent(x_bin_index)
            #print 'Adding bin content: ', hist.GetBinContent(x_bin_index)
            combined_bckg_hist.AddBinContent(x_bin_index,hist.GetBinContent(x_bin_index))
        if cumulative_bckg_entries!=0 and tmp_bin_content_ttHww/cumulative_bckg_entries>=0.02:
            new_x_bin_edge = hist_list.get('ttW').GetXaxis().GetBinUpEdge(x_bin_index)
            x_bin_edges.append(new_x_bin_edge)
            cumulative_bckg_entries = 0


    '''N_total = combined_bckg_hist.Integral() + combined_sig_hist.Integral()
    if N_total == 0:
        print 'Warning! Integral of sig + hist combined = 0'
        nYQ = [0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.6125,0.6875,0.75,0.8125,0.875,0.9375,1.0]
    else:
        Bin = int(math.floor(N_total/5.))
        XQ = np.zeros(Bin)
        YQ = np.zeros(Bin)
        nYQ = np.zeros(Bin+1)
        for i in xrange(0,Bin):
            XQ[i]=(i+1.)/Bin
        combined_bckg_hist.GetQuantiles(Bin, YQ, XQ) #now YQ contains the low bin edge
        for i in xrange(0,Bin):
            nYQ[i+1]=YQ[i] #shift YQ

    n_xbins = len(nYQ)-1
    x_bin_edges_fuckyouroot = array('d',nYQ)'''


    #x_bin_edges.append(1.)
    #ordered_bin_edges = x_bin_edges
    ordered_bin_edges = [0.,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.6125,0.6875,0.75,0.8125,0.875,0.9375,1.0]
    n_xbins = len(ordered_bin_edges)-1
    x_bin_edges_fuckyouroot = array('d',ordered_bin_edges)

    new_hists = {}

    print 'hist_list:'
    print hist_list

    for hist_key, hist in hist_list.iteritems():
        print 'hist_key: ', hist_key
        print 'n_xbins: ', n_xbins
        print 'hist.GetName(): ', hist.GetName()
        print 'x_bin_edges_fuckyouroot: ', x_bin_edges_fuckyouroot

        #if hist.Integral() == 0:
        #    continue
        hist_list[hist_key] = hist.Rebin(n_xbins, hist.GetName(), x_bin_edges_fuckyouroot)
        for x_bin_index in xrange(0,hist_list[hist_key].GetNbinsX()):
            tmp_bin_content = hist_list[hist_key].GetBinContent(x_bin_index)

    #hist_list['Data'] = data_hist.Rebin(n_xbins,data_hist.GetName(),x_bin_edges_fuckyouroot)
    return hist_list, x_bin_edges_fuckyouroot


# Makes plots of the various categories for the DNN. Each 'category' histogram is a plot of the response on one of the DNN nodes.
# However, it contains only the events that showed the highest DNN response for said node.
def make_plot(stacked_hist, category, norm, legend, inputs_directory, separation_, option_, data_hist = None, ratioplot = None):

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

    classifier_plots_dir = os.path.join('samples_w_DNN',inputs_directory,'plots')
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
    l1.SetNDC()
    l1.SetTextFont(43)
    l1.SetTextSize(15)
    l1.SetTextAlign(12)
    latex_title = ''
    if 'ttHCategory' in category:
        latex_title = "#it{prefit: ttH Category}"
    elif 'ttWCategory' in category:
        latex_title = "#it{prefit: ttW Category}"
    elif 'OtherCategory' in category:
        latex_title = "#it{prefit: Other Category}"
    elif 'tHqCategory' in category:
        latex_title = "#it{prefit: tHq Category}"

    l1.DrawLatex(0.15,0.8,latex_title)

    l2=ROOT.TLatex()
    l2.SetNDC()
    l2.SetTextFont(43)
    l2.SetTextSize(12)
    separation_title = 'Separation = %s' % ("{0:.5g}".format(separation_))
    l2.DrawLatex(0.15,0.75,separation_title)

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

    outfile_name = 'DNNOutput_%s_%s_%s.pdf'%(category,norm_suffix,option_)
    output_fullpath = classifier_plots_dir + '/' + outfile_name
    c1.Print(output_fullpath,'pdf')

def separation_table(outputdir,separation_dictionary):
    content = r'''\documentclass{article}
\begin{document}
\begin{center}
\begin{table}
\begin{tabular}{| c | c | c | c | c |} \hline
Option \textbackslash Node & ttH & ttJ & ttW & ttZ \\ \hline
%s \\
\hline
\end{tabular}
\caption{Separation power on each output node. The separation is given with respect to the `signal' process the node is trained to separate (one node per column) and the combined background processes for that node. The three options represent the different mehtods to of class weights in the DNN training.}
\end{table}
\end{center}
\end{document}
'''
    table_path = os.path.join(outputdir,'separation_table')
    table_tex = table_path+'.tex'
    print 'table_tex: ', table_tex
    with open(table_tex,'w') as f:
        option_1_entry = '%s & %s & %s & %s & %s' % ('Option 1', "{0:.5g}".format(separation_dictionary['option1'][0]), "{0:.5g}".format(separation_dictionary['option1'][1]), "{0:.5g}".format(separation_dictionary['option1'][2]), "{0:.5g}".format(separation_dictionary['option1'][3]))
        #option_2_entry = '%s & %s & %s & %s & %s' % ('Option 2', "{0:.5g}".format(separation_dictionary['option2'][0]), "{0:.5g}".format(separation_dictionary['option2'][1]), "{0:.5g}".format(separation_dictionary['option2'][2]), "{0:.5g}".format(separation_dictionary['option2'][3]))
        #option_3_entry = '%s & %s & %s & %s & %s' % ('Option 3', "{0:.5g}".format(separation_dictionary['option3'][0]), "{0:.5g}".format(separation_dictionary['option3'][1]), "{0:.5g}".format(separation_dictionary['option3'][2]), "{0:.5g}".format(separation_dictionary['option3'][3]))
        #option_4_entry = '%s & %s & %s & %s & %s' % ('Option 4', "{0:.5g}".format(separation_dictionary['option4'][0]), "{0:.5g}".format(separation_dictionary['option4'][1]), "{0:.5g}".format(separation_dictionary['option4'][2]), "{0:.5g}".format(separation_dictionary['option4'][3]))
        #f.write( content % (option_1_entry, option_2_entry, option_3_entry) )
        f.write( content % (option_1_entry) )
    return

def main():
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-d', '--data',        dest='data_flag'  ,      help='1 = include data from plots, 0 = exclude data from plots', default=0, type=int)
    parser.add_argument('-r', '--region', dest='region', help='Option to choose DiLepRegion', default='DiLepRegion', type=str)
    parser.add_argument('-m', '--model_dir', dest='model_dir', help='Option to choose directory containing model. Choose directory from samples_w_DNN', default='', type=str)
    args = parser.parse_args()

    data_flag = args.data_flag
    region = args.region

    inputs_directory = args.model_dir

    classifier_samples_dir = os.path.join("samples_w_DNN/",inputs_directory)

    print 'Reading samples from: ', classifier_samples_dir

    #sample_name = ['Convs','EWK','Fakes','Flips','Rares','TTH_hww','TTH_htt','TTH_hzz','TTW','TTZ','TTWW','TTH_hmm','Data','TTH_hzz']
    sample_name = ['Convs','EWK','Fakes','Flips','Rares','TTH_hww','TTH_htt','TTW','TTZ','TTWW','TTH_hmm','TTH_hzz','THQ_hww','THQ_htt','THQ_hzz']
    channel_name = ['2017samples_tH_tunedweights_tH']

    input_name_Conv = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[0],region)
    input_file_Conv = TFile.Open(input_name_Conv)
    input_name_EWK = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[1],region)
    input_file_EWK = TFile.Open(input_name_EWK)
    input_name_Fakes = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[2],region)
    input_file_Fakes = TFile.Open(input_name_Fakes)
    input_name_Flips = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[3],region)
    input_file_Flips = TFile.Open(input_name_Flips)
    input_name_Rares = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[4],region)
    input_file_Rares = TFile.Open(input_name_Rares)
    input_name_TTH_hww = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[5],region)
    input_file_TTH_hww = TFile.Open(input_name_TTH_hww)
    input_name_TTH_htt = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[6],region)
    input_file_TTH_htt = TFile.Open(input_name_TTH_htt)
    input_name_TTW = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[7],region)
    input_file_TTW = TFile.Open(input_name_TTW)
    input_name_TTZ = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[8],region)
    input_file_TTZ = TFile.Open(input_name_TTZ)
    input_name_TTWW = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[9],region)
    input_file_TTWW = TFile.Open(input_name_TTWW)
    input_name_TTH_hmm = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[10],region)
    input_file_TTH_hmm = TFile.Open(input_name_TTH_hmm)
    input_name_TTH_hzz = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[11],region)
    input_file_TTH_hzz = TFile.Open(input_name_TTH_hzz)
    input_name_tHq_hww = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[12],region)
    input_file_tHq_hww = TFile.Open(input_name_tHq_hww)
    input_name_tHq_htt = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[13],region)
    input_file_tHq_htt = TFile.Open(input_name_tHq_htt)
    input_name_tHq_hzz = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[14],region)
    input_file_tHq_hzz = TFile.Open(input_name_tHq_hzz)
    #input_name_data = '%s/%s/%s_%s.root' % (classifier_samples_dir,channel_name[0],sample_name[12],region)
    #input_file_data = TFile.Open(input_name_data)

    categories = ['ttHCategory','OtherCategory','ttWCategory','tHqCategory']

    separation_dictionary = OrderedDict([
        ('option1' , [])
    ])

    for cat in categories:

        process_list = OrderedDict([
        ("Conv" , 5),
        ("EWK" , 6),
        ("Fakes" , 13),
        ("Flips" , 17),
        ("Rares" , 38),
        ("ttW" , 8),
        ("ttZ" , 9),
        ("ttWW" , 3),
        ("ttH_HWW" , 2),
        ("ttH_HZZ" , 2),
        ("ttH_Htautau" , 2),
        ("ttH_Hmm" , 2),
        ("tHq_HWW" , 2),
        ("tHq_HZZ" , 2),
        ("tHq_Htautau" , 2),
        ("Data" , 1)
        ])

        option_name = ['option1']

        #bckg_hists = ROOT.TH1F('bckg_hists','bckg_hists',n_xbins,x_bin_edges_fuckyouroot)
        #sig_hists = ROOT.TH1F('sig_hists','sig_hists',n_xbins,x_bin_edges_fuckyouroot)

        for option_ in option_name:
            print option_
            hist_stack = ROOT.THStack()

            histo_Conv_name = 'histo_%s_events_Conv' % (cat)
            histo_EWK_name = 'histo_%s_events_EWK' % (cat)
            histo_Fakes_name = 'histo_%s_events_Fakes' % (cat)
            histo_Flips_name = 'histo_%s_events_Flips' % (cat)
            histo_Rares_name = 'histo_%s_events_Rares' % (cat)
            histo_TTH_hww_name = 'histo_%s_events_ttH_HWW' % (cat)
            histo_TTH_htt_name = 'histo_%s_events_ttH_Htautau' % (cat)
            histo_TTH_hmm_name = 'histo_%s_events_ttH_Hmm' % (cat)
            histo_TTH_hzz_name = 'histo_%s_events_ttH_HZZ' % (cat)

            histo_tHq_hww_name = 'histo_%s_events_tHq_HWW' % (cat)
            histo_tHq_htt_name = 'histo_%s_events_tHq_Htautau' % (cat)
            histo_tHq_hzz_name = 'histo_%s_events_tHq_HZZ' % (cat)

            histo_TTW_name = 'histo_%s_events_ttW' % (cat)
            histo_TTZ_name = 'histo_%s_events_ttZ' % (cat)
            histo_TTWW_name = 'histo_%s_events_ttWW' % (cat)
            histo_data_name = 'histo_%s_events_Data' % (cat)

            histo_Conv_sample = input_file_Conv.Get(histo_Conv_name)
            histo_EWK_sample = input_file_EWK.Get(histo_EWK_name)
            histo_Fakes_sample = input_file_Fakes.Get(histo_Fakes_name)
            histo_Flips_sample = input_file_Flips.Get(histo_Flips_name)
            histo_Rares_sample = input_file_Rares.Get(histo_Rares_name)
            histo_TTH_hww_sample = input_file_TTH_hww.Get(histo_TTH_hww_name)
            histo_TTH_htt_sample = input_file_TTH_htt.Get(histo_TTH_htt_name)
            histo_TTH_hmm_sample = input_file_TTH_hmm.Get(histo_TTH_hmm_name)
            histo_TTH_hzz_sample = input_file_TTH_hzz.Get(histo_TTH_hzz_name)

            histo_tHq_hww_sample = input_file_tHq_hww.Get(histo_tHq_hww_name)
            histo_tHq_htt_sample = input_file_tHq_htt.Get(histo_tHq_htt_name)
            histo_tHq_hzz_sample = input_file_tHq_hzz.Get(histo_tHq_hzz_name)

            histo_TTW_sample = input_file_TTW.Get(histo_TTW_name)
            histo_TTZ_sample = input_file_TTZ.Get(histo_TTZ_name)
            histo_TTWW_sample = input_file_TTWW.Get(histo_TTWW_name)
            #histo_data_sample = input_file_data.Get(histo_data_name)

            # Turn this into dictionary
            hist_list = OrderedDict([
            ("Conv" , histo_Conv_sample),
            ("EWK" , histo_EWK_sample),
            ("Fakes" , histo_Fakes_sample),
            ("Flips" , histo_Flips_sample),
            ("Rares" , histo_Rares_sample),
            ("ttW" , histo_TTW_sample),
            ("ttZ" , histo_TTZ_sample),
            ("ttWW" , histo_TTWW_sample),
            ("ttH_HWW" , histo_TTH_hww_sample),
            ("ttH_Htautau" , histo_TTH_htt_sample),
            ("ttH_Hmm" , histo_TTH_hmm_sample),
            ("ttH_HZZ" , histo_TTH_hzz_sample),
            ("tHq_HWW" , histo_tHq_hww_sample),
            ("tHq_Htautau" , histo_tHq_htt_sample),
            ("tHq_HZZ" , histo_tHq_hzz_sample)
            #("Data" , histo_data_sample)
            ])

            #nBins = hist_list.get('ttW').GetNbinsX()
            # Rebin Histograms so > 0 total background entries per bin.
            # Returns array of histograms in same order as was passed to the function.

            #rebinned_histograms, x_bin_edges_ = rebinHistograms(hist_list, histo_data_sample)
            rebinned_histograms, x_bin_edges_ = rebinHistograms(hist_list)
            bckg_hists = ROOT.TH1F('bckg_hists','bckg_hists',len(x_bin_edges_)-1,x_bin_edges_)
            sig_hists = ROOT.TH1F('sig_hists','sig_hists',len(x_bin_edges_)-1,x_bin_edges_)
            print 'sig_hists; ', sig_hists
            legend = TLegend(0.7,  0.7,  0.9,  0.9)
            legend.SetNColumns(2)

            if cat == 'ttHCategory':
                signal_string = 'ttH_HWW,ttH_Htautau,ttH_Hmm,ttH_HZZ'
            if cat == 'OtherCategory':
                signal_string = 'Fakes,Flips,Conv,ttZ'
            if cat == 'ttWCategory':
                signal_string = 'ttW'
            elif cat == 'tHqCategory':
                signal_string = 'tHq'

            #rebincount = 0
            for rebinned_hist_name, rebinned_hist in rebinned_histograms.iteritems():
                #if rebincount == 0:
                #    rebinned_hist.Sumw2()
                #rebincount = 1
                rebinned_hist.SetMarkerColor(process_list[rebinned_hist_name])
                rebinned_hist.SetLineColor(process_list[rebinned_hist_name])
                rebinned_hist.GetYaxis().SetTitle("Events/bin")
                rebinned_hist.SetMarkerStyle(20)
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
                    if 'TTH_hzz' in rebinned_hist_name:
                        rebinned_hist.SetFillStyle(6001)
                    else:
                        rebinned_hist.SetFillStyle(3001)
                    rebinned_hist.SetFillColor(process_list[rebinned_hist_name])
                    hist_stack.Add(rebinned_hist)
                    if rebinned_hist_name not in signal_string:
                        bckg_hists.Add(rebinned_hist)
                    if rebinned_hist_name in signal_string:
                        if 'ttWW' == rebinned_hist_name:
                            bckg_hists.Add(rebinned_hist)
                        else:
                            sig_hists.Add(rebinned_hist)

            hist_stack.SetMaximum(hist_stack.GetStack().Last().GetMaximum() + (hist_stack.GetStack().Last().GetMaximum()/2))
            hist_stack.SetMinimum(0.)

            separation_ =  GetSeparation(sig_hists,bckg_hists)
            separation_dictionary[option_].append(separation_)

            # Get Data/MC agreement
            # Ensure to pass data histogram as last argument.
            if data_flag:
                dataOverMC = GetDataOverMC(hist_stack,rebinned_histograms['Data'])
                # Create histograms.
                make_plot(hist_stack, cat, False, legend, inputs_directory, separation_, option_, rebinned_histograms['Data'] , dataOverMC)
            else:
                make_plot(hist_stack, cat, False, legend, inputs_directory, separation_, option_)

            bckg_hists.Reset()
            sig_hists.Reset()
        hist_list.clear()

    #for key, value in separation_dictionary.iteritems():
    #print 'saving table to ', classifier_samples_dir
    separation_table(classifier_samples_dir,separation_dictionary)
main()
