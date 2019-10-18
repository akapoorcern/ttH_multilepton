from plotting.control_plotter import control_plotter
from collections import OrderedDict
import optparse, json, argparse
import os
def main():
    CPlotter = control_plotter()
    output_dir = 'control_plots_2019-07-26_newfiles/'

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)

    parser.add_argument('-s', '--sel', dest='selection', help='Option to choose selection', default='geq3j', type=str)

    args = parser.parse_args()
    selection = args.selection

    CPlotter.check_dir(output_dir)
    files_list = []
    files_list_training = []

    input_path_ttH_HWW_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/TTH_hww_DiLepRegion.root')[0]
    input_path_ttH_HZZ_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/TTH_hzz_DiLepRegion.root')[0]
    input_path_ttH_Htautau_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/TTH_htt_DiLepRegion.root')[0]
    input_path_ttH_Hmm_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/TTH_hmm_DiLepRegion.root')[0]
    input_path_ttH_Hother_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/TTH_hot_DiLepRegion.root')[0]
    input_path_Conv_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/Conv_DiLepRegion.root')[0]
    input_path_Fakes_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/Fakes_DiLepRegion.root')[0]
    input_path_Flips_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/Flips_DiLepRegion.root')[0]
    input_path_ttW_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/TTW_DiLepRegion.root')[0]
    input_path_ttZ_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/TTZ_DiLepRegion.root')[0]
    input_path_ttWW_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/TTWW_DiLepRegion.root')[0]
    input_path_EWK_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/EWK_DiLepRegion.root')[0]
    input_path_data_DiLepRegion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/Rootplas_WithTH_20190629/DiLepRegion/Data_DiLepRegion.root')[0]

    files_list.append(input_path_Conv_DiLepRegion)
    files_list.append(input_path_Fakes_DiLepRegion)
    files_list.append(input_path_Flips_DiLepRegion)
    files_list.append(input_path_ttW_DiLepRegion)
    files_list.append(input_path_ttZ_DiLepRegion)
    files_list.append(input_path_ttH_HWW_DiLepRegion)
    files_list.append(input_path_ttH_HZZ_DiLepRegion)
    files_list.append(input_path_ttH_Htautau_DiLepRegion)
    files_list.append(input_path_ttH_Hmm_DiLepRegion)
    files_list.append(input_path_ttH_Hother_DiLepRegion)
    files_list.append(input_path_ttWW_DiLepRegion)
    files_list.append(input_path_EWK_DiLepRegion)
    files_list.append(input_path_data_DiLepRegion)

    files_ = CPlotter.open_files(files_list)

    branch_list = [
    #'n_presel_jet',
    #'nBJetLoose',
    #'nBJetMedium',
    #'jet1_pt',
    #'jet1_eta',
    #'jet1_phi',
    #'jet1_E',
    #'jet2_pt',
    #'jet2_eta',
    #'jet2_phi',
    #'jet2_E',
    #'jet3_pt',
    #'jet3_eta',
    #'jet3_phi',
    #'jet3_E',
    #'jet4_pt',
    #'jet4_eta',
    #'jet4_phi',
    #'jet4_E',
    #'lep1_conePt',
    #'lep1_eta',
    #'lep1_phi',
    #'lep1_E',
    #'lep2_conePt',
    #'lep2_eta',
    #'lep2_phi',
    #'lep2_E',
    #'hadTop_BDT',
    #'Hj_tagger_hadTop',
    #'resTop_BDT',
    #'Hj_tagger_resTop',
    #'metLD',
    #'maxeta',
    'massL'
    #'mindr_lep1_jet',
    #'mindr_lep2_jet',
    #'avg_dr_jet',
    #'mT_lep1',
    #'mT_lep2',
    #'mbb',
    #'n_presel_ele',
    #'n_presel_mu',
    #'Dilep_pdgId',
    #'lep1_charge',
    #'jetFwd1_pt',
    #'jetFwd1_eta',
    #'n_presel_jetFwd'
    ]

    # Start building histogram names you wish to plot (check 'load_histograms' in plotting class).
    # Can simply use the elements of previous dictionary or plottings sum hists class to create summed histogram.

    file_ttH_HWW_DiLepRegion_keyname = input_path_ttH_HWW_DiLepRegion.split('/')[-1:]
    file_ttH_HWW_DiLepRegion_keyname = file_ttH_HWW_DiLepRegion_keyname[0].split('.')[:-1]
    file_ttH_HZZ_DiLepRegion_keyname = input_path_ttH_HZZ_DiLepRegion.split('/')[-1:]
    file_ttH_HZZ_DiLepRegion_keyname = file_ttH_HZZ_DiLepRegion_keyname[0].split('.')[:-1]
    file_ttH_Htautau_DiLepRegion_keyname = input_path_ttH_Htautau_DiLepRegion.split('/')[-1:]
    file_ttH_Htautau_DiLepRegion_keyname = file_ttH_Htautau_DiLepRegion_keyname[0].split('.')[:-1]
    file_ttH_Hmm_DiLepRegion_keyname = input_path_ttH_Hmm_DiLepRegion.split('/')[-1:]
    file_ttH_Hmm_DiLepRegion_keyname = file_ttH_Hmm_DiLepRegion_keyname[0].split('.')[:-1]
    file_ttH_Hoth_DiLepRegion_keyname = input_path_ttH_Hother_DiLepRegion.split('/')[-1:]
    file_ttH_Hoth_DiLepRegion_keyname = file_ttH_Hoth_DiLepRegion_keyname[0].split('.')[:-1]
    file_Conv_DiLepRegion_keyname = input_path_Conv_DiLepRegion.split('/')[-1:]
    file_Conv_DiLepRegion_keyname = file_Conv_DiLepRegion_keyname[0].split('.')[:-1]
    file_Fakes_DiLepRegion_keyname = input_path_Fakes_DiLepRegion.split('/')[-1:]
    file_Fakes_DiLepRegion_keyname = file_Fakes_DiLepRegion_keyname[0].split('.')[:-1]
    file_Flips_DiLepRegion_keyname = input_path_Flips_DiLepRegion.split('/')[-1:]
    file_Flips_DiLepRegion_keyname = file_Flips_DiLepRegion_keyname[0].split('.')[:-1]
    file_ttW_DiLepRegion_keyname = input_path_ttW_DiLepRegion.split('/')[-1:]
    file_ttW_DiLepRegion_keyname = file_ttW_DiLepRegion_keyname[0].split('.')[:-1]
    file_ttZ_DiLepRegion_keyname = input_path_ttZ_DiLepRegion.split('/')[-1:]
    file_ttZ_DiLepRegion_keyname = file_ttZ_DiLepRegion_keyname[0].split('.')[:-1]
    file_ttWW_DiLepRegion_keyname = input_path_ttWW_DiLepRegion.split('/')[-1:]
    file_ttWW_DiLepRegion_keyname = file_ttWW_DiLepRegion_keyname[0].split('.')[:-1]
    file_EWK_DiLepRegion_keyname = input_path_EWK_DiLepRegion.split('/')[-1:]
    file_EWK_DiLepRegion_keyname = file_EWK_DiLepRegion_keyname[0].split('.')[:-1]
    file_data_DiLepRegion_keyname = input_path_data_DiLepRegion.split('/')[-1:]
    file_data_DiLepRegion_keyname = file_data_DiLepRegion_keyname[0].split('.')[:-1]

    files_histname_dict = OrderedDict()
    for branch_index in xrange(len(branch_list)):
        branch_to_draw = []
        branch_to_draw.append(branch_list[branch_index])
        print '<main> Draw branch: ', branch_to_draw
        files_histname_dict = CPlotter.load_histos(files_, branch_to_draw, 'syncTree', selection)

        file_ttH_HWW_key = file_ttH_HWW_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_ttH_HZZ_key = file_ttH_HZZ_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_ttH_Htautau_key = file_ttH_Htautau_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_ttH_Hmm_key = file_ttH_Hmm_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_ttH_Hoth_key = file_ttH_Hoth_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_Conv_key = file_Conv_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_Fakes_key = file_Fakes_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_Flips_key = file_Flips_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_ttW_DiLepRegion_key = file_ttW_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_ttWW_DiLepRegion_key = file_ttWW_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_EWK_DiLepRegion_key = file_EWK_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_ttZ_DiLepRegion_key = file_ttZ_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_data_DiLepRegion_key = file_data_DiLepRegion_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"

        combined_hist_MC_SR_title = branch_list[branch_index]+'_combined_MC'

        output_file_name = branch_list[branch_index] + '_DataMC.png'
        output_fullpath = os.path.join(output_dir, output_file_name)
        CPlotter.output_fullpath = output_fullpath

        data_hist_name = 'Data_DiLepRegion_%s_DiLepRegion' % (branch_list[branch_index])
        input_hist_data = files_histname_dict.get(data_hist_name)

        CPlotter.stack_hists(files_histname_dict, combined_hist_MC_SR_title, branch_list[branch_index], input_hist_data)
        files_histname_dict.clear()
        print '<main> output in ', output_fullpath

    exit(0)

main()
