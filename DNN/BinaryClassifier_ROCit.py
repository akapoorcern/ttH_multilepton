import ROOT
from ROOT import TFile, TTree, gDirectory, gROOT, TH1, TF1, TProfile, TProfile2D, TLegend
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
import optparse


def make_roc_dist(MVA_signal_histo, MVA_bckg_histo, MVA_ROC_sig_eff, MVA_ROC_bckg_rej):

    nBins = MVA_signal_histo.GetXaxis().GetNbins()

    lowedge_cutval = -999
    dnn_cuts = []
    for i in xrange(1,MVA_signal_histo.GetNbinsX()):
        lowedge_cutval = MVA_signal_histo.GetXaxis().GetBinLowEdge(i)
        dnn_cuts.append(lowedge_cutval)

    for i in xrange(len(dnn_cuts)):
        signal_passing_cut = 0
        signal_total = 0
        bckg1_passing_cut = 0
        bckg1_total = 0
        temp_cut_val = dnn_cuts[i]
        for j in xrange(MVA_signal_histo.GetNbinsX()):
            signal_lowedge = MVA_signal_histo.GetXaxis().GetBinLowEdge(j)
            signal_bin_content = MVA_signal_histo.GetBinContent(j)
            signal_total = signal_total + signal_bin_content
            signal_passing_cut =  signal_passing_cut+signal_bin_content if signal_lowedge >= temp_cut_val else signal_passing_cut

            bckg1_lowedge = MVA_bckg_histo.GetXaxis().GetBinLowEdge(j)
            bckg1_bin_content = MVA_bckg_histo.GetBinContent(j)
            bckg1_total = bckg1_total + bckg1_bin_content
            bckg1_passing_cut = bckg1_passing_cut+bckg1_bin_content if bckg1_lowedge >= temp_cut_val else bckg1_passing_cut

        true_positive_rate = signal_passing_cut / signal_total
        bckg_rej = 1 - (bckg1_passing_cut/bckg1_total)
        MVA_ROC_sig_eff.append(true_positive_rate)
        MVA_ROC_bckg_rej.append(bckg_rej)

'''
def histogram_settings():

def print_canvas():
'''


def main():

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify inputs from network training',      default=None,        type='string')
    (opt, args) = parser.parse_args()

    classifier_suffix = opt.input_suffix

    MVA_list = [('DNN','red'),('BDTG','blue')]
    plt.figure(1)
    title_label = 'Binary Classifier ROC Curves'
    plt.title(title_label)
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.grid(True)
    plt.xlabel('Signal Eff.')
    plt.ylabel('Bckg. Rej.')
    legend = plt.legend(prop={'size': 8})
    plt.subplots_adjust(right=0.8)
    auc_text_box=''
    dirty_counter = 0
    for MVA_label, colour in MVA_list:
        classifier_parent_dir = 'BinaryClassifier_%s_%s' % (MVA_label, classifier_suffix)
        print 'classifier_parent_dir: ', classifier_parent_dir
        classifier_plots_dir = classifier_parent_dir+"/plots"
        if not os.path.exists(classifier_plots_dir):
            os.makedirs(classifier_plots_dir)
        input_name = '%s/outputs/Applied_%s.root' % (classifier_parent_dir,classifier_parent_dir)
        print 'input_name: ' , input_name
        input_root = TFile.Open(input_name)

        if 'ttJets' in classifier_suffix:
            bckg_name = 'ttjets'
        elif 'ttV' in classifier_suffix:
            bckg_name = 'ttV'

        applied_ttH_name = "histo_response_ttH"
        applied_bckg_name = "histo_response_%s" % (bckg_name)
        ttH_histo = input_root.Get(applied_ttH_name)
        bckg_histo = input_root.Get(applied_bckg_name)
        applied_MVA_ROC_sig_eff = []
        applied_MVA_ROC_bckg_rej = []
        make_roc_dist(ttH_histo, bckg_histo, applied_MVA_ROC_sig_eff, applied_MVA_ROC_bckg_rej)
        x_ROC_sig_eff = np.array(applied_MVA_ROC_sig_eff)
        y_ROC_bckg_rej = np.array(applied_MVA_ROC_bckg_rej)
        # Using trapezoidal rule as approximation for integral.
        area_ROC_bckg_rej = trapz(x_ROC_sig_eff, y_ROC_bckg_rej, dx=(1./40.))

        plt.plot(applied_MVA_ROC_sig_eff,applied_MVA_ROC_bckg_rej, color=colour, label=MVA_label)

        if dirty_counter == len(MVA_list)-1:
            auc_text = '%s AUC = %s' % (MVA_label,format(area_ROC_bckg_rej, '.2f'))
        else:
            auc_text = '%s AUC = %s \n' % (MVA_label,format(area_ROC_bckg_rej, '.2f'))
        auc_text_box += auc_text
        dirty_counter = dirty_counter + 1
        #plt.subplots_adjust(right=0.8)
        #plt.savefig(classifier_parent_dir+'/plots/BDTG_ROC_apply.png')
    plt.figtext(0.7, 0.7, auc_text_box, wrap=True, horizontalalignment='center', fontsize=8, bbox=dict(fc="none"))
    figure_name = 'BinaryClassifier_%s_ROCs.png' % classifier_suffix
    plt.savefig(figure_name)



main()
