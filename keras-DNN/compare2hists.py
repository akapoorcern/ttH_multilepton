from plotting.control_plotter import control_plotter
from collections import OrderedDict
import optparse, json, argparse
import os
def main():
    CPlotter = control_plotter()
    output_dir = 'compare_TRSR_newsamples2017/'

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)

    parser.add_argument('-s', '--sel', dest='selection', help='Option to choose selection', default='tH', type=str)

    args = parser.parse_args()
    selection = args.selection

    CPlotter.check_dir(output_dir)
    files_list = []
    files_list_training = []

    #input_path_1 = CPlotter.getEOSlsfile(directory='/b/binghuan/Rootplas/Legacy/rootplas_LegacyMVA_1113/DiLepRegion/ttH2017TrainDNN2L/ttHnobb_DiLepRegion.root')[0]
    #input_path_2 = CPlotter.getEOSlsfile(directory='/b/binghuan/Rootplas/Legacy/rootplas_LegacyAll_1110/DiLepRegion/2017/DiLepRegion/TTH_hww_ctcvcp_DiLepRegion.root')[0]

    #input_path_1 = CPlotter.getEOSlsfile(directory='/b/binghuan/Rootplas/Legacy/rootplas_LegacyMVA_1113/DiLepRegion/ttH2017TrainDNN2L/ttWJets_DiLepRegion.root')[0]
    #input_path_2 = CPlotter.getEOSlsfile(directory='/b/binghuan/Rootplas/Legacy/rootplas_LegacyAll_1110/DiLepRegion/2017/DiLepRegion/TTW_DiLepRegion.root')[0]

    #input_path_1 = CPlotter.getEOSlsfile(directory='/b/binghuan/Rootplas/Legacy/rootplas_LegacyMVA_1113/DiLepRegion/ttH2017TrainDNN2L/ttZJets_DiLepRegion.root')[0]
    #input_path_2 = CPlotter.getEOSlsfile(directory='/b/binghuan/Rootplas/Legacy/rootplas_LegacyAll_1110/DiLepRegion/2017/DiLepRegion/TTZ_DiLepRegion.root')[0]


    #input_path_1 = CPlotter.getEOSlsfile(directory='/b/binghuan/Rootplas/TrainMVA_looseWithTH_20190628/ttJets_DiLepRegion.root')[0]
    #input_path_2 = CPlotter.getEOSlsfile(directory='/j/jthomasw/ttH2L/samples/FakesFlips_oldsamples.root')[0]

    input_path_1 = CPlotter.getEOSlsfile(directory='/b/binghuan/Rootplas/Legacy/rootplas_LegacyMVA_1113/DiLepRegion/ttH2017TrainDNN2L/ttJets_PS_DiLepRegion.root')[0]
    input_path_2 = CPlotter.getEOSlsfile(directory='/j/jthomasw/ttH2L/samples/FakesFlips_newest2017SR.root')[0]

    #input_path_1 = CPlotter.getEOSlsfile(directory='/j/jthomasw/ttH2L/samples/FakesFlips_oldsamples.root')[0]
    #input_path_2 = CPlotter.getEOSlsfile(directory='/j/jthomasw/ttH2L/samples/FakesFlips_2017SignalRegion.root')[0]

    files_list.append(input_path_1)
    files_list.append(input_path_2)

    files_ = CPlotter.open_files(files_list)

    branch_list = [
    'n_presel_jet',
    'nBJetLoose',
    'nBJetMedium',
    'n_presel_jetFwd',
    #'jet1_pt',
    #'jet1_eta',
    #'jet1_phi',
    #'jet2_pt',
    #'jet2_eta',
    #'jet2_phi',
    #'jet3_pt',
    #'jet3_eta',
    #'jet3_phi',
    #'jet4_pt',
    #'jet4_eta',
    #'jet4_phi',
    'lep1_conePt',
    #'lep1_eta',
    #'lep1_phi',
    #'lep2_conePt',
    #'lep2_eta',
    #'lep2_phi',
    #'hadTop_BDT',
    #'Hj_tagger_hadTop',
    #'metLD',
    #'maxeta',
    #'mass_dilep',
    #'mindr_lep1_jet',
    #'mindr_lep2_jet',
    #'avg_dr_jet',
    #'mT_lep1',
    #'mT_lep2',
    #'mbb',
    #'Dilep_pdgId',
    #'lep1_charge',
    #'jetFwd1_pt',
    #'jetFwd1_eta'
    ]

    # Start building histogram names you wish to plot (check 'load_histograms' in plotting class).
    # Can simply use the elements of previous dictionary or plottings sum hists class to create summed histogram.

    file_1_keyname = input_path_1.split('/')[-1:]
    file_1_keyname = file_1_keyname[0].split('.')[:-1]
    file_2_keyname = input_path_2.split('/')[-1:]
    file_2_keyname = file_2_keyname[0].split('.')[:-1]

    process = ''
    if 'ttJets' in file_1_keyname[0]:
        process = 'ttJets'
    elif 'ttW' in file_1_keyname[0]:
        process = 'ttW'
    elif 'ttZ' in file_1_keyname[0]:
        process = 'ttZ'
    elif 'tHQ' in file_1_keyname[0]:
        process = 'tHQ'
    elif 'ttH' in file_1_keyname[0]:
        process = 'ttH'
    elif 'Other' in file_1_keyname[0]:
        process = 'Other'

    print 'Comparing file %s with %s' % (file_1_keyname, file_2_keyname)

    files_histname_dict = OrderedDict()
    for branch_index in xrange(len(branch_list)):
        branch_to_draw = []
        branch_to_draw.append(branch_list[branch_index])
        print '<main> Draw branch: ', branch_to_draw
        files_histname_dict = CPlotter.load_histos(files_, branch_to_draw, 'syncTree', selection)

        file_1_key = file_1_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"
        file_2_key = file_2_keyname[0] + '_' + branch_list[branch_index] + "_DiLepRegion"

        output_file_name = file_1_keyname[0] + '_' + branch_list[branch_index] + '_TRSR_comparison.png'
        output_fullpath = os.path.join(output_dir, output_file_name)
        CPlotter.output_fullpath = output_fullpath

        print 'Getting hist: %s and hist: %s' % (file_1_key,file_2_key)

        hist1_ = files_histname_dict.get(file_1_key)
        hist2_ = files_histname_dict.get(file_2_key)

        title1=process+' SR old'
        title2=process+' SR new'

        CPlotter.make_comparison(hist1_, title1 , hist2_, title2, branch_list[branch_index], output_fullpath)
        files_histname_dict.clear()
        print '<main> output in ', output_fullpath

    exit(0)

main()
