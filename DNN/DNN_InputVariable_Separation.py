#################################
#     DNN_ResponsePlotter.py
#     Joshuha Thomas-Wilsker
#       IHEP Bejing, CERN
#################################
# Plotting script using output
# root files from TMVA training.
#################################

import ROOT
from ROOT import TFile, TTree, gDirectory, gROOT, TH1, TF1, TProfile, TProfile2D, TLegend, TLatex
import os
import optparse
from scipy import stats
import numpy as np
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

    for i in xrange(nBins):
        sig_bin_norm = hist_sig.GetBinContent(i)/nS
        bckg_bin_norm = hist_bckg.GetBinContent(i)/nB
        # Separation:
        if(sig_bin_norm+bckg_bin_norm > 0):
            separation += 0.5 * ((sig_bin_norm - bckg_bin_norm) * (sig_bin_norm - bckg_bin_norm)) / (sig_bin_norm + bckg_bin_norm)
    separation *= dX
    return separation

def plot_input_variable(input_root, classifier_suffix, variable_transformation, variable_name, json_data):
    classifier_parent_dir = 'MultiClass_DNN_%s' % (classifier_suffix)
    variable_roofile_ttHsample_histogram_name = '%s/InputVariables_%s/%s__ttH_%s' % (classifier_parent_dir,variable_transformation,variable_name,variable_transformation)
    variable_roofile_ttVsample_histogram_name = '%s/InputVariables_%s/%s__ttV_%s' % (classifier_parent_dir,variable_transformation,variable_name,variable_transformation)
    variable_roofile_ttJetssample_histogram_name = '%s/InputVariables_%s/%s__ttJets_%s' % (classifier_parent_dir,variable_transformation,variable_name,variable_transformation)

    histo_inputvar_ttH = input_root.Get(str(variable_roofile_ttHsample_histogram_name))
    histo_inputvar_ttV = input_root.Get(str(variable_roofile_ttVsample_histogram_name))
    histo_inputvar_ttJets = input_root.Get(str(variable_roofile_ttJetssample_histogram_name))

    maxyaxis = max(histo_inputvar_ttH.GetMaximum(), histo_inputvar_ttV.GetMaximum(), histo_inputvar_ttJets.GetMaximum())
    histo_inputvar_ttH.SetAxisRange(0.,maxyaxis+1,"Y")

    histo_inputvar_ttH.Sumw2()
    histo_inputvar_ttV.Sumw2()
    histo_inputvar_ttJets.Sumw2()

    c1 = ROOT.TCanvas("c1","c1",900,700)
    p1 = ROOT.TPad("p1","p1", 0.0,0.0,1.0,1.0)
    p1.Draw()
    p1.SetBottomMargin(0.1)
    p1.SetTopMargin(0.1)
    p1.SetLeftMargin(0.1)
    p1.SetRightMargin(0.1)
    p1.SetGridx(True)
    p1.SetGridy(True)
    p1.cd()
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(0)

    histo_inputvar_ttH.SetLineColor(2)
    histo_inputvar_ttH.SetMarkerColor(2)
    histo_inputvar_ttH.SetMarkerStyle(20)

    histo_inputvar_ttV.SetLineColor(3)
    histo_inputvar_ttV.SetMarkerColor(3)
    histo_inputvar_ttV.SetMarkerStyle(20)

    histo_inputvar_ttJets.SetLineColor(4)
    histo_inputvar_ttJets.SetMarkerColor(4)
    histo_inputvar_ttJets.SetMarkerStyle(20)

    histo_inputvar_ttH.GetYaxis().SetTitle("(1/N)dN/dX")

    legend = TLegend(0.8,  0.7,  0.99,  0.99)
    legend.AddEntry(histo_inputvar_ttH,"ttH events")
    legend.AddEntry(histo_inputvar_ttV,"ttV events")
    legend.AddEntry(histo_inputvar_ttJets,"tt+jets")

    histo_inputvar_ttH.Draw('EP')
    histo_inputvar_ttV.Draw('EPsame')
    histo_inputvar_ttJets.Draw('EPsame')

    # Add custom title
    l1=ROOT.TLatex()
    l1.SetNDC();
    histo_title = "%s" % variable_name
    l1.DrawLatex(0.36,0.94,histo_title)


    sep_ttH_v_ttV = GetSeparation(histo_inputvar_ttH, histo_inputvar_ttV)
    sep_ttH_v_ttJ = GetSeparation(histo_inputvar_ttH, histo_inputvar_ttJets)
    latex_separation_ttH_v_ttV = '#scale[0.5]{ttH vs. ttV separation = %.5f}' % sep_ttH_v_ttV
    latex_separation_ttH_v_ttJets = '#scale[0.5]{ttH vs. ttJ separation = %.5f}' % sep_ttH_v_ttJ
    json_separation_ttH_v_ttV = '{%s : ttH vs. ttV separation : %s}' % (variable_name,sep_ttH_v_ttV)
    json_separation_ttH_v_ttJets = '{%s : ttH vs. ttJ separation : %s}' % (variable_name,sep_ttH_v_ttV)

    sep_ttV_v_ttH = GetSeparation(histo_inputvar_ttV, histo_inputvar_ttH)
    sep_ttV_v_ttJets = GetSeparation(histo_inputvar_ttV, histo_inputvar_ttJets)
    latex_separation_ttV_v_ttH = '#scale[0.5]{ttV vs. ttH separation = %.5f}' % sep_ttV_v_ttH
    latex_separation_ttV_v_ttJets = '#scale[0.5]{ttV vs. ttJ separation = %.5f}' % sep_ttV_v_ttJets

    sep_ttJets_v_ttH = GetSeparation(histo_inputvar_ttJets, histo_inputvar_ttH)
    sep_ttJets_v_ttV = GetSeparation(histo_inputvar_ttJets, histo_inputvar_ttV)
    latex_separation_ttJets_v_ttH = '#scale[0.5]{ttJ vs. ttH separation = %.5f}' % sep_ttJets_v_ttH
    latex_separation_ttJets_v_ttV = '#scale[0.5]{ttJ vs. ttV separation = %.5f}' % sep_ttJets_v_ttV

    json_data[variable_name] = []
    json_data[variable_name].append({
    'ttH vs. ttV separation':sep_ttH_v_ttV,
    'ttH vs. ttJ separation':sep_ttH_v_ttJ,
    'ttV vs. ttH separation':sep_ttV_v_ttH,
    'ttV vs. ttJ separation':sep_ttV_v_ttJets,
    'ttJ vs. ttH separation':sep_ttJets_v_ttH,
    'ttJ vs. ttV separation':sep_ttJets_v_ttV,
    })

    l2=ROOT.TLatex()
    l2.SetNDC();
    l2.DrawLatex(0.65,0.65,latex_separation_ttH_v_ttV)

    l3=ROOT.TLatex()
    l3.SetNDC();
    l3.DrawLatex(0.65,0.6,latex_separation_ttH_v_ttJets)

    l3=ROOT.TLatex()
    l3.SetNDC();
    l3.DrawLatex(0.65,0.55,latex_separation_ttV_v_ttH)

    l3=ROOT.TLatex()
    l3.SetNDC();
    l3.DrawLatex(0.65,0.5,latex_separation_ttV_v_ttJets)

    l3=ROOT.TLatex()
    l3.SetNDC();
    l3.DrawLatex(0.65,0.45,latex_separation_ttJets_v_ttH)

    l3=ROOT.TLatex()
    l3.SetNDC();
    l3.DrawLatex(0.65,0.4,latex_separation_ttJets_v_ttV)

    legend.Draw("sameP")

    c1.cd()
    c1.Modified()
    c1.Update()

    outfile_name = '%s/plots/MCDNN_InputVariable_%s-%s.pdf' % (classifier_parent_dir, variable_transformation, variable_name)
    c1.Print(outfile_name,'pdf')
    c1.Clear()

def main():
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-s', '--suffix',        dest='input_suffix'  ,      help='suffix used to identify inputs from network training',      default=None,        type='string')
    parser.add_option('-j', '--json',        dest='json'  ,      help='json file with list of variables',      default=None,        type='string')

    (opt, args) = parser.parse_args()

    if opt.input_suffix == None:
        print 'Input files suffix not defined!'
        sys.exit(1)
    if opt.json == None:
        print 'Input variables json not defined!'
        sys.exit(1)

    classifier_suffix = opt.input_suffix
    jsonFile = open(opt.json,'r')
    new_variable_list = json.load(jsonFile,encoding='utf-8').items()

    classifier_parent_dir = 'MultiClass_DNN_%s' % (classifier_suffix)
    classifier_plots_dir = classifier_parent_dir+"/plots"
    if not os.path.exists(classifier_plots_dir):
        os.makedirs(classifier_plots_dir)

    classifier_samples_dir = classifier_parent_dir+"/outputs"

    input_name = '%s/MultiClass_DNN_%s.root' % (classifier_samples_dir,classifier_suffix)
    input_root = TFile.Open(input_name)


    output_data_dir =  classifier_parent_dir + "/data"
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    output_json_name = '%s/input_variable_separations.json' % (output_data_dir)

    if 'D+G-VarTrans' in classifier_suffix:
        variable_transformation_name = 'Deco_Gauss'
    if 'D-VarTrans' in classifier_suffix:
        variable_transformation_name = 'Deco'

    with open(output_json_name,'w') as output_json_file:
        for key, value in new_variable_list:
            if 'hadTop_BDT' in key:
                keyname = 'hadTop_BDT'
            elif 'Hj1_BDT' in key:
                keyname = 'Hj1_BDT'
            else:
                keyname = key
            print 'variable: ', key
            json_data = {}
            plot_input_variable(input_root, classifier_suffix, variable_transformation_name, keyname, json_data)
            json.dump(json_data,output_json_file)
            output_json_file.write("\n")


main()
