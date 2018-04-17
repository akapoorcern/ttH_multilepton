import ROOT
from ROOT import TFile, TTree, gDirectory, gROOT, TH1, TF1, TProfile, TProfile2D, TLegend
import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz

def make_roc_dist(DNN_test_signal, DNN_test_bckg1, DNN_test_bckg2, dnn_ROC_sig_eff, dnn_ROC_bckg_rej):

    nBins = DNN_test_signal.GetXaxis().GetNbins()

    lowedge_cutval = -999
    dnn_cuts = []
    for i in xrange(DNN_test_signal.GetNbinsX()):
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

    classifier_suffix = '2HLs_relu'
    classifier_parent_dir = 'MultiClass_DNN_%s' % (classifier_suffix)
    classifier_plots_dir = classifier_parent_dir+"/plots"

    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    input_name = 'ttHML_MCDNN_%s.root' % (classifier_suffix)

    print 'input_name: ' , input_name

    input_root = TFile.Open(input_name)
    DNN_rejBvsS_ttHnode_test_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_rejBvsS_ttH" % (classifier_suffix)
    DNN_rejBvsS_ttVnode_test_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_rejBvsS_ttV" % (classifier_suffix)
    DNN_rejBvsS_ttJetsnode_test_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_rejBvsS_ttJets" % (classifier_suffix)
    DNN_rejBvsS_ttHnode_train_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_rejBvsS_ttH" % (classifier_suffix)
    DNN_rejBvsS_ttVnode_train_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_rejBvsS_ttV" % (classifier_suffix)
    DNN_rejBvsS_ttJetsnode_train_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_rejBvsS_ttJets" % (classifier_suffix)

    DNN_ttHnode_test_ttH_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttH" % (classifier_suffix)
    DNN_ttHnode_test_ttV_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttV" % (classifier_suffix)
    DNN_ttHnode_test_ttJets_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_ttH_prob_for_ttJets" % (classifier_suffix)

    DNN_ttVnode_test_ttH_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttH" % (classifier_suffix)
    DNN_ttVnode_test_ttV_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttV" % (classifier_suffix)
    DNN_ttVnode_test_ttJets_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_ttV_prob_for_ttJets" % (classifier_suffix)

    DNN_ttJetsnode_test_ttH_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttH" % (classifier_suffix)
    DNN_ttJetsnode_test_ttV_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttV" % (classifier_suffix)
    DNN_ttJetsnode_test_ttJets_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Test_ttJets_prob_for_ttJets" % (classifier_suffix)

    DNN_ttHnode_train_ttH_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttH" % (classifier_suffix)
    DNN_ttHnode_train_ttV_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttV" % (classifier_suffix)
    DNN_ttHnode_train_ttJets_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_ttH_prob_for_ttJets" % (classifier_suffix)

    DNN_ttVnode_train_ttH_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttH" % (classifier_suffix)
    DNN_ttVnode_train_ttV_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttV" % (classifier_suffix)
    DNN_ttVnode_train_ttJets_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_ttV_prob_for_ttJets" % (classifier_suffix)

    DNN_ttJetsnode_train_ttH_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttH" % (classifier_suffix)
    DNN_ttJetsnode_train_ttV_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttV" % (classifier_suffix)
    DNN_ttJetsnode_train_ttJets_name = "MultiClass_DNN_%s/Method_DNN/DNN/MVA_DNN_Train_ttJets_prob_for_ttJets" % (classifier_suffix)

    DNN_rejBvsS_ttHnode_test = input_root.Get(DNN_rejBvsS_ttHnode_test_name)
    DNN_rejBvsS_ttVnode_test = input_root.Get(DNN_rejBvsS_ttVnode_test_name)
    DNN_rejBvsS_ttJetsnode_test = input_root.Get(DNN_rejBvsS_ttJetsnode_test_name)
    DNN_rejBvsS_ttHnode_train = input_root.Get(DNN_rejBvsS_ttHnode_train_name)
    DNN_rejBvsS_ttVnode_train = input_root.Get(DNN_rejBvsS_ttVnode_train_name)
    DNN_rejBvsS_ttJetsnode_train = input_root.Get(DNN_rejBvsS_ttJetsnode_train_name)

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

    x_ROC_sig_eff_ttVnode = np.array(train_dnn_ROC_sig_eff_ttVnode)
    y_ROC_bckg_rej_ttVnode = np.array(train_dnn_ROC_bckg_rej_ttVnode)

    x_ROC_sig_eff_ttJetsnode = np.array(train_dnn_ROC_sig_eff_ttJetsnode)
    y_ROC_bckg_rej_ttJetsnode = np.array(train_dnn_ROC_bckg_rej_ttJetsnode)

    # Using trapezoidal rule as approximation for integral.
    area_ROC_bckg_rej_ttHnode = trapz(x_ROC_sig_eff_ttHnode, y_ROC_bckg_rej_ttHnode, dx=(1./40.))
    area_ROC_bckg_rej_ttVnode = trapz(x_ROC_sig_eff_ttVnode, y_ROC_bckg_rej_ttVnode, dx=(1./40.))
    area_ROC_bckg_rej_ttJetsnode = trapz(x_ROC_sig_eff_ttJetsnode, y_ROC_bckg_rej_ttJetsnode, dx=(1./40.))

    plt.figure(1)
    plt.plot(test_dnn_ROC_sig_eff_ttHnode,test_dnn_ROC_bckg_rej_ttHnode, color='k', label='ttH')
    plt.plot(test_dnn_ROC_sig_eff_ttVnode,test_dnn_ROC_bckg_rej_ttVnode, color='g', label='ttV')
    plt.plot(test_dnn_ROC_sig_eff_ttJetsnode,test_dnn_ROC_bckg_rej_ttJetsnode, color='b', label='tt+jets')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.grid(True)
    plt.xlabel('Signal Eff.')
    plt.ylabel('Bckg. Rej.')
    legend = plt.legend()

    ttHnode_auc_text = 'ttH node AUC = %s' % format(area_ROC_bckg_rej_ttHnode, '.2f')
    ttVnode_auc_text = 'ttV node AUC = %s' % format(area_ROC_bckg_rej_ttVnode, '.2f')
    ttJetsnode_auc_text = 'ttjets node AUC = %s' % format(area_ROC_bckg_rej_ttJetsnode, '.2f')

    auc_text_box = ttHnode_auc_text + "\n" + ttVnode_auc_text + "\n" + ttJetsnode_auc_text

    plt.figtext(0.9, 0.9, auc_text_box, wrap=True, horizontalalignment='center', fontsize=8, bbox=dict(fc="none"))
    #plt.subplots_adjust(left=0.1)
    plt.subplots_adjust(right=0.8)
    plt.savefig(classifier_parent_dir+'/plots/DNN_ROC_test.png')

    plt.figure(2)
    plt.plot(train_dnn_ROC_sig_eff_ttHnode,train_dnn_ROC_bckg_rej_ttHnode, color='k', label='ttH')
    plt.plot(train_dnn_ROC_sig_eff_ttVnode,train_dnn_ROC_bckg_rej_ttVnode, color='g', label='ttV')
    plt.plot(train_dnn_ROC_sig_eff_ttJetsnode,train_dnn_ROC_bckg_rej_ttJetsnode, color='b', label='tt+jets')
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.grid(True)
    plt.xlabel('Signal Eff.')
    plt.ylabel('Bckg. Rej.')
    legend = plt.legend()
    plt.savefig(classifier_parent_dir+'/plots/DNN_ROC_train.png')

    DNN_ROC_canvas_test = ROOT.TCanvas("c1","c1",900,700)
    DNN_ROC_pad_test = ROOT.TPad("p1","p1",0,0,1,1)
    DNN_ROC_pad_test.Draw()
    DNN_ROC_pad_test.SetBottomMargin(0.1)
    DNN_ROC_pad_test.SetTopMargin(0.1)
    DNN_ROC_pad_test.SetLeftMargin(0.1)
    DNN_ROC_pad_test.SetRightMargin(0.1)
    DNN_ROC_pad_test.SetGridx(True)
    DNN_ROC_pad_test.SetGridy(True)
    DNN_ROC_pad_test.cd()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    DNN_rejBvsS_ttHnode_test.SetLineColor(2)
    DNN_rejBvsS_ttVnode_test.SetLineColor(3)
    DNN_rejBvsS_ttJetsnode_test.SetLineColor(4)

    DNN_rejBvsS_ttHnode_test.GetXaxis().SetTitle('Signal Efficiency')
    DNN_rejBvsS_ttHnode_test.GetYaxis().SetTitle('Background Rejection (1-FPR)')

    DNN_rejBvsS_ttHnode_test.Draw()
    DNN_rejBvsS_ttVnode_test.Draw('same')
    DNN_rejBvsS_ttJetsnode_test.Draw('same')

    legend = TLegend(0.8, 0.7, 0.99, 0.99)
    legend.AddEntry(DNN_rejBvsS_ttHnode_test, "ttH node", "l")
    legend.AddEntry(DNN_rejBvsS_ttVnode_test, "ttV node", "l")
    legend.AddEntry(DNN_rejBvsS_ttJetsnode_test, "ttJets node", "l")
    legend.Draw("same")

    DNN_ROC_canvas_test.cd()
    DNN_ROC_canvas_test.Modified()
    DNN_ROC_canvas_test.Update()

    outfile_name = '%s/plots/MCDNN_ROCs_%s.pdf' % (classifier_parent_dir, classifier_suffix)
    DNN_ROC_canvas_test.Print(outfile_name,'pdf')
    DNN_ROC_canvas_test.Clear()

    #make_roc_dist()



main()
