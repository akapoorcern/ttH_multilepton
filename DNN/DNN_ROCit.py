import ROOT
from ROOT import TFile, TTree, gDirectory, gROOT, TH1, TF1, TProfile, TProfile2D, TLegend
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
import optparse


def make_roc_dist(DNN_test_signal, DNN_test_bckg1, DNN_test_bckg2, dnn_ROC_sig_eff, dnn_ROC_bckg_rej):

    nBins = DNN_test_signal.GetXaxis().GetNbins()

    lowedge_cutval = -999
    dnn_cuts = []
    for i in xrange(1,DNN_test_signal.GetNbinsX()):
        lowedge_cutval = DNN_test_signal.GetXaxis().GetBinLowEdge(i)
        dnn_cuts.append(lowedge_cutval)

    for i in xrange(len(dnn_cuts)):
        signal_passing_cut = 0
        signal_total = 0
        bckg1_passing_cut = 0
        bckg1_total = 0
        bckg2_passing_cut = 0
        bckg2_total = 0
        temp_cut_val = dnn_cuts[i]
        for j in xrange(DNN_test_signal.GetNbinsX()):
            signal_lowedge = DNN_test_signal.GetXaxis().GetBinLowEdge(j)
            signal_bin_content = DNN_test_signal.GetBinContent(j)
            signal_total = signal_total + signal_bin_content
            signal_passing_cut =  signal_passing_cut+signal_bin_content if signal_lowedge >= temp_cut_val else signal_passing_cut

            bckg1_lowedge = DNN_test_bckg1.GetXaxis().GetBinLowEdge(j)
            bckg1_bin_content = DNN_test_bckg1.GetBinContent(j)
            bckg1_total = bckg1_total + bckg1_bin_content
            bckg1_passing_cut = bckg1_passing_cut+bckg1_bin_content if bckg1_lowedge >= temp_cut_val else bckg1_passing_cut

            bckg2_lowedge = DNN_test_bckg2.GetXaxis().GetBinLowEdge(j)
            bckg2_bin_content = DNN_test_bckg2.GetBinContent(j)
            bckg2_total = bckg2_total + bckg2_bin_content
            bckg2_passing_cut = bckg2_passing_cut + bckg2_bin_content if bckg2_lowedge>= temp_cut_val else bckg2_passing_cut

        true_positive_rate = signal_passing_cut / signal_total
        bckg_rej = 1 - ((bckg1_passing_cut+bckg2_passing_cut)/(bckg1_total + bckg2_total))
        dnn_ROC_sig_eff.append(true_positive_rate)
        dnn_ROC_bckg_rej.append(bckg_rej)

'''
def histogram_settings():

def print_canvas():
'''
def main():

    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify inputs from network training',      default='2HLs_relu',        type='string')
    (opt, args) = parser.parse_args()

    classifier_suffix = opt.input_suffix

    classifier_parent_dir = 'V7-DNN_%s' % (classifier_suffix)
    classifier_plots_dir = classifier_parent_dir+"/plots"

    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    input_name = '%s/outputs/%s.root' % (classifier_parent_dir,classifier_parent_dir)

    print 'input_name: ' , input_name

    input_root = TFile.Open(input_name)

    DNN_ttHnode_test_ttH_name = "%s/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttH" % (classifier_parent_dir)
    DNN_ttHnode_test_ttV_name = "%s/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttV" % (classifier_parent_dir)
    DNN_ttHnode_test_ttJets_name = "%s/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttJets" % (classifier_parent_dir)

    DNN_ttVnode_test_ttH_name = "%s/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttH" % (classifier_parent_dir)
    DNN_ttVnode_test_ttV_name = "%s/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttV" % (classifier_parent_dir)
    DNN_ttVnode_test_ttJets_name = "%s/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttJets" % (classifier_parent_dir)

    DNN_ttJetsnode_test_ttH_name = "%s/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttH" % (classifier_parent_dir)
    DNN_ttJetsnode_test_ttV_name = "%s/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttV" % (classifier_parent_dir)
    DNN_ttJetsnode_test_ttJets_name = "%s/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttJets" % (classifier_parent_dir)

    DNN_ttHnode_train_ttH_name = "%s/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttH" % (classifier_parent_dir)
    DNN_ttHnode_train_ttV_name = "%s/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttV" % (classifier_parent_dir)
    DNN_ttHnode_train_ttJets_name = "%s/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttJets" % (classifier_parent_dir)

    DNN_ttVnode_train_ttH_name = "%s/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttH" % (classifier_parent_dir)
    DNN_ttVnode_train_ttV_name = "%s/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttV" % (classifier_parent_dir)
    DNN_ttVnode_train_ttJets_name = "%s/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttJets" % (classifier_parent_dir)

    DNN_ttJetsnode_train_ttH_name = "%s/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttH" % (classifier_parent_dir)
    DNN_ttJetsnode_train_ttV_name = "%s/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttV" % (classifier_parent_dir)
    DNN_ttJetsnode_train_ttJets_name = "%s/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttJets" % (classifier_parent_dir)


    DNN_ttHnode_test_ttH = input_root.Get(DNN_ttHnode_test_ttH_name)
    DNN_ttHnode_test_ttV = input_root.Get(DNN_ttHnode_test_ttV_name)
    DNN_ttHnode_test_ttJets = input_root.Get(DNN_ttHnode_test_ttJets_name)

    DNN_ttVnode_test_ttH = input_root.Get(DNN_ttVnode_test_ttH_name)
    DNN_ttVnode_test_ttV = input_root.Get(DNN_ttVnode_test_ttV_name)
    DNN_ttVnode_test_ttJets = input_root.Get(DNN_ttVnode_test_ttJets_name)

    DNN_ttJetsnode_test_ttH = input_root.Get(DNN_ttJetsnode_test_ttH_name)
    DNN_ttJetsnode_test_ttV = input_root.Get(DNN_ttJetsnode_test_ttV_name)
    DNN_ttJetsnode_test_ttJets = input_root.Get(DNN_ttJetsnode_test_ttJets_name)

    DNN_ttHnode_train_ttH = input_root.Get(DNN_ttHnode_train_ttH_name)
    DNN_ttHnode_train_ttV = input_root.Get(DNN_ttHnode_train_ttV_name)
    DNN_ttHnode_train_ttJets = input_root.Get(DNN_ttHnode_train_ttJets_name)

    DNN_ttVnode_train_ttH = input_root.Get(DNN_ttVnode_train_ttH_name)
    DNN_ttVnode_train_ttV = input_root.Get(DNN_ttVnode_train_ttV_name)
    DNN_ttVnode_train_ttJets = input_root.Get(DNN_ttVnode_train_ttJets_name)

    DNN_ttJetsnode_train_ttH = input_root.Get(DNN_ttJetsnode_train_ttH_name)
    DNN_ttJetsnode_train_ttV = input_root.Get(DNN_ttJetsnode_train_ttV_name)
    DNN_ttJetsnode_train_ttJets = input_root.Get(DNN_ttJetsnode_train_ttJets_name)

    test_dnn_ROC_sig_eff_ttHnode = []
    test_dnn_ROC_bckg_rej_ttHnode = []
    test_dnn_ROC_sig_eff_ttVnode = []
    test_dnn_ROC_bckg_rej_ttVnode = []
    test_dnn_ROC_sig_eff_ttJetsnode = []
    test_dnn_ROC_bckg_rej_ttJetsnode = []
    make_roc_dist(DNN_ttHnode_test_ttH, DNN_ttHnode_test_ttV, DNN_ttHnode_test_ttJets, test_dnn_ROC_sig_eff_ttHnode, test_dnn_ROC_bckg_rej_ttHnode)
    make_roc_dist(DNN_ttVnode_test_ttV, DNN_ttVnode_test_ttH, DNN_ttVnode_test_ttJets, test_dnn_ROC_sig_eff_ttVnode, test_dnn_ROC_bckg_rej_ttVnode)
    make_roc_dist(DNN_ttJetsnode_test_ttJets, DNN_ttJetsnode_test_ttH, DNN_ttJetsnode_test_ttV, test_dnn_ROC_sig_eff_ttJetsnode, test_dnn_ROC_bckg_rej_ttJetsnode)

    train_dnn_ROC_sig_eff_ttHnode = []
    train_dnn_ROC_bckg_rej_ttHnode = []
    train_dnn_ROC_sig_eff_ttVnode = []
    train_dnn_ROC_bckg_rej_ttVnode = []
    train_dnn_ROC_sig_eff_ttJetsnode = []
    train_dnn_ROC_bckg_rej_ttJetsnode = []
    make_roc_dist(DNN_ttHnode_train_ttH, DNN_ttHnode_train_ttV, DNN_ttHnode_train_ttJets, train_dnn_ROC_sig_eff_ttHnode, train_dnn_ROC_bckg_rej_ttHnode)
    make_roc_dist(DNN_ttVnode_train_ttV, DNN_ttVnode_train_ttH, DNN_ttVnode_train_ttJets, train_dnn_ROC_sig_eff_ttVnode, train_dnn_ROC_bckg_rej_ttVnode)
    make_roc_dist(DNN_ttJetsnode_train_ttJets, DNN_ttJetsnode_train_ttH, DNN_ttJetsnode_train_ttV, train_dnn_ROC_sig_eff_ttJetsnode, train_dnn_ROC_bckg_rej_ttJetsnode)

    x_ROC_sig_eff_ttHnode = np.array(test_dnn_ROC_sig_eff_ttHnode)
    y_ROC_bckg_rej_ttHnode = np.array(test_dnn_ROC_bckg_rej_ttHnode)

    x_ROC_sig_eff_ttVnode = np.array(test_dnn_ROC_sig_eff_ttVnode)
    y_ROC_bckg_rej_ttVnode = np.array(test_dnn_ROC_bckg_rej_ttVnode)

    x_ROC_sig_eff_ttJetsnode = np.array(test_dnn_ROC_sig_eff_ttJetsnode)
    y_ROC_bckg_rej_ttJetsnode = np.array(test_dnn_ROC_bckg_rej_ttJetsnode)

    # Using trapezoidal rule as approximation for integral.
    area_ROC_bckg_rej_ttHnode = trapz(x_ROC_sig_eff_ttHnode, y_ROC_bckg_rej_ttHnode, dx=(1./40.))
    area_ROC_bckg_rej_ttVnode = trapz(x_ROC_sig_eff_ttVnode, y_ROC_bckg_rej_ttVnode, dx=(1./40.))
    area_ROC_bckg_rej_ttJetsnode = trapz(x_ROC_sig_eff_ttJetsnode, y_ROC_bckg_rej_ttJetsnode, dx=(1./40.))

    plt.figure(1)
    plt.title('DNN Output Node ROC Curves: test')
    plt.plot(test_dnn_ROC_sig_eff_ttHnode,test_dnn_ROC_bckg_rej_ttHnode, color='k', label='ttH node')
    plt.plot(test_dnn_ROC_sig_eff_ttVnode,test_dnn_ROC_bckg_rej_ttVnode, color='g', label='ttV node')
    plt.plot(test_dnn_ROC_sig_eff_ttJetsnode,test_dnn_ROC_bckg_rej_ttJetsnode, color='b', label='tt+jets node')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.grid(True)
    plt.xlabel('Signal Eff.')
    plt.ylabel('Bckg. Rej.')
    legend = plt.legend(prop={'size': 8})

    ttHnode_auc_text = 'ttH node AUC = %s' % format(area_ROC_bckg_rej_ttHnode, '.2f')
    ttVnode_auc_text = 'ttV node AUC = %s' % format(area_ROC_bckg_rej_ttVnode, '.2f')
    ttJetsnode_auc_text = 'ttjets node AUC = %s' % format(area_ROC_bckg_rej_ttJetsnode, '.2f')

    auc_text_box = ttHnode_auc_text + "\n" + ttVnode_auc_text + "\n" + ttJetsnode_auc_text

    plt.figtext(0.75, 0.6, auc_text_box, wrap=True, horizontalalignment='center', fontsize=8, bbox=dict(fc="none"))
    #plt.subplots_adjust(left=0.1)
    #plt.subplots_adjust(right=0.8)
    ROC_fig_test_name = '%s/plots/%s_ROC_test.png' % (classifier_parent_dir,classifier_parent_dir)
    plt.savefig(ROC_fig_test_name)

    plt.figure(2)
    plt.title('DNN Output Node ROC Curves: train')
    plt.plot(train_dnn_ROC_sig_eff_ttHnode,train_dnn_ROC_bckg_rej_ttHnode, color='k', label='ttH node')
    plt.plot(train_dnn_ROC_sig_eff_ttVnode,train_dnn_ROC_bckg_rej_ttVnode, color='g', label='ttV node')
    plt.plot(train_dnn_ROC_sig_eff_ttJetsnode,train_dnn_ROC_bckg_rej_ttJetsnode, color='b', label='tt+jets node')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.grid(True)
    plt.xlabel('Signal Eff.')
    plt.ylabel('Bckg. Rej.')
    legend = plt.legend(prop={'size': 8})

    x_ROC_sig_eff_ttHnode_train = np.array(train_dnn_ROC_sig_eff_ttHnode)
    y_ROC_bckg_rej_ttHnode_train = np.array(train_dnn_ROC_bckg_rej_ttHnode)

    x_ROC_sig_eff_ttVnode_train = np.array(train_dnn_ROC_sig_eff_ttVnode)
    y_ROC_bckg_rej_ttVnode_train = np.array(train_dnn_ROC_bckg_rej_ttVnode)

    x_ROC_sig_eff_ttJetsnode_train = np.array(train_dnn_ROC_sig_eff_ttJetsnode)
    y_ROC_bckg_rej_ttJetsnode_train = np.array(train_dnn_ROC_bckg_rej_ttJetsnode)

    area_ROC_bckg_rej_ttHnode_train = trapz(x_ROC_sig_eff_ttHnode_train, y_ROC_bckg_rej_ttHnode_train, dx=(1./40.))
    area_ROC_bckg_rej_ttVnode_train = trapz(x_ROC_sig_eff_ttVnode_train, y_ROC_bckg_rej_ttVnode_train, dx=(1./40.))
    area_ROC_bckg_rej_ttJetsnode_train = trapz(x_ROC_sig_eff_ttJetsnode_train, y_ROC_bckg_rej_ttJetsnode_train, dx=(1./40.))

    ttHnode_auc_text_train = 'ttH node AUC = %s' % format(area_ROC_bckg_rej_ttHnode_train, '.2f')
    ttVnode_auc_text_train = 'ttV node AUC = %s' % format(area_ROC_bckg_rej_ttVnode_train, '.2f')
    ttJetsnode_auc_text_train = 'ttjets node AUC = %s' % format(area_ROC_bckg_rej_ttJetsnode_train, '.2f')

    auc_text_box_train = ttHnode_auc_text_train + "\n" + ttVnode_auc_text_train + "\n" + ttJetsnode_auc_text_train

    plt.figtext(0.75, 0.6, auc_text_box_train, wrap=True, horizontalalignment='center', fontsize=8, bbox=dict(fc="none"))

    ROC_fig_train_name = '%s/plots/%s_ROC_train.png' % (classifier_parent_dir,classifier_parent_dir)
    plt.savefig(ROC_fig_train_name)

main()
