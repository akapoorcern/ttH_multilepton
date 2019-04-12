from plotting.control_plotter import control_plotter
from collections import OrderedDict

def main():
    CPlotter = control_plotter()
    CPlotter.check_dir('invar_control_plots_190404')
    files_list = []
    input_path_ttH_loose = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190315/loose/NoJetNCut/ttHnobb_NoJetNCut.root')[0]
    input_path_ttH_fakeable = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190315/fakeable/NoJetNCut/ttHnobb_NoJetNCut.root')[0]
    input_path_ttH_HWW_signalregion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/TTH_hww_SigRegion.root')[0]
    input_path_ttH_HZZ = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/TTH_hzz_SigRegion.root')[0]
    input_path_ttH_Htautau = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/TTH_htt_SigRegion.root')[0]
    input_path_ttH_Hmm = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/TTH_hmm_SigRegion.root')[0]
    input_path_ttH_Hother = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/TTH_hot_SigRegion.root')[0]

    input_path_ttJ_training = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190315/loose/NoJetNCut/ttJets_NoJetNCut.root')[0]
    input_path_ttJ_fakeable = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190315/fakeable/NoJetNCut/ttJets_NoJetNCut.root')[0]
    input_path_Conv_signalregion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/Conv_SigRegion.root')[0]
    input_path_Fakes_signalregion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/Fakes_SigRegion.root')[0]
    input_path_Flips_signalregion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/Flips_SigRegion.root')[0]

    input_path_ttW_training = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190315/loose/NoJetNCut/ttWJets_NoJetNCut.root')[0]
    input_path_ttW_fakeable = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190315/fakeable/NoJetNCut/ttWJets_NoJetNCut.root')[0]
    input_path_ttW_signalregion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/TTW_SigRegion.root')[0]

    input_path_ttZ_training = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190315/loose/NoJetNCut/ttZJets_NoJetNCut.root')[0]
    input_path_ttZ_fakeable = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190315/fakeable/NoJetNCut/ttZJets_NoJetNCut.root')[0]
    input_path_ttZ_signalregion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/TTZ_SigRegion.root')[0]

    input_path_ttWW_signalregion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/TTWW_SigRegion.root')[0]
    input_path_EWK_signalregion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/EWK_SigRegion.root')[0]

    #input_path_data_loose = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190315/loose/NoJetNCut/ttZJets_NoJetNCut.root')[0]
    #input_path_data_fakeable = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190315/fakeable/NoJetNCut/ttZJets_NoJetNCut.root')[0]
    input_path_data_signalregion = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/Data_SigRegion.root')[0]

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
    ("ttH_other" , 2),
    ("ttH_Hmm" , 2),
    ("Data" , 1)
    ])


    files_list.append(input_path_Conv_signalregion)
    files_list.append(input_path_Fakes_signalregion)
    files_list.append(input_path_Flips_signalregion)
    files_list.append(input_path_ttW_signalregion)
    files_list.append(input_path_ttZ_signalregion)
    files_list.append(input_path_ttH_HWW_signalregion)
    files_list.append(input_path_ttH_HZZ)
    files_list.append(input_path_ttH_Htautau)
    files_list.append(input_path_ttH_Hmm)
    files_list.append(input_path_ttH_Hother)
    files_list.append(input_path_ttWW_signalregion)
    files_list.append(input_path_EWK_signalregion)
    files_list.append(input_path_data_signalregion)

    files_ = CPlotter.open_files(files_list)

    '''branch_list = [
    'nBJetMedium',
    'jet1_pt',
    'jet1_eta',
    'jet2_pt',
    'jet2_eta',
    'jet3_pt',
    'jet3_eta',
    'jet4_pt',
    'jet4_eta',
    'lep1_conePt',
    'lep1_eta',
    'lep2_conePt',
    'lep2_eta',
    'resTop_BDT',
    'Hj_tagger_resTop',
    'metLD',
    'maxeta',
    'massL',
    'mindr_lep1_jet',
    'mindr_lep2_jet',
    'avg_dr_jet',
    'mT_lep1',
    'mT_lep2',
    'mbb',
    'n_presel_ele',
    'n_presel_mu',
    ]'''

    branch_list = [
    'nLepFO',
    'lep1_charge',
    'Dilep_pdgId',
    ]

    #files_histname_dict = CPlotter.load_histos(files_,branch_list,'syncTree')

    # Start building histogram names you wish to plot (check 'load_histograms' in plotting class).
    # Can simply use the elements of previous dictionary or plottings sum hists class to create summed histogram.
    file_ttH_training_keyname = input_path_ttH_loose.split('/')[-1:]
    file_ttH_training_keyname = file_ttH_training_keyname[0].split('.')[:-1]
    file_ttH_HWW_keyname = input_path_ttH_HWW_signalregion.split('/')[-1:]
    file_ttH_HWW_keyname = file_ttH_HWW_keyname[0].split('.')[:-1]
    file_ttH_HZZ_keyname = input_path_ttH_HZZ.split('/')[-1:]
    file_ttH_HZZ_keyname = file_ttH_HZZ_keyname[0].split('.')[:-1]
    file_ttH_Htautau_keyname = input_path_ttH_Htautau.split('/')[-1:]
    file_ttH_Htautau_keyname = file_ttH_Htautau_keyname[0].split('.')[:-1]
    file_ttH_Hmm_keyname = input_path_ttH_Hmm.split('/')[-1:]
    file_ttH_Hmm_keyname = file_ttH_Hmm_keyname[0].split('.')[:-1]
    file_ttH_Hoth_keyname = input_path_ttH_Hother.split('/')[-1:]
    file_ttH_Hoth_keyname = file_ttH_Hoth_keyname[0].split('.')[:-1]
    file_ttH_fakeable_keyname = input_path_ttH_fakeable.split('/')[-1:]
    file_ttH_fakeable_keyname = file_ttH_fakeable_keyname[0].split('.')[:-1]

    file_ttJ_training_keyname = input_path_ttJ_training.split('/')[-1:]
    file_ttJ_training_keyname = file_ttJ_training_keyname[0].split('.')[:-1]
    file_ttJ_fakeable_keyname = input_path_ttJ_fakeable.split('/')[-1:]
    file_ttJ_fakeable_keyname = file_ttJ_fakeable_keyname[0].split('.')[:-1]
    file_Conv_keyname = input_path_Conv_signalregion.split('/')[-1:]
    file_Conv_keyname = file_Conv_keyname[0].split('.')[:-1]
    file_Fakes_keyname = input_path_Fakes_signalregion.split('/')[-1:]
    file_Fakes_keyname = file_Fakes_keyname[0].split('.')[:-1]
    file_Flips_keyname = input_path_Flips_signalregion.split('/')[-1:]
    file_Flips_keyname = file_Flips_keyname[0].split('.')[:-1]

    file_ttW_training_keyname = input_path_ttW_training.split('/')[-1:]
    file_ttW_training_keyname = file_ttW_training_keyname[0].split('.')[:-1]
    file_ttW_signalregion_keyname = input_path_ttW_signalregion.split('/')[-1:]
    file_ttW_signalregion_keyname = file_ttW_signalregion_keyname[0].split('.')[:-1]
    file_ttW_fakeable_keyname = input_path_ttW_fakeable.split('/')[-1:]
    file_ttW_fakeable_keyname = file_ttW_fakeable_keyname[0].split('.')[:-1]

    file_ttZ_training_keyname = input_path_ttZ_training.split('/')[-1:]
    file_ttZ_training_keyname = file_ttZ_training_keyname[0].split('.')[:-1]
    file_ttZ_signalregion_keyname = input_path_ttZ_signalregion.split('/')[-1:]
    file_ttZ_signalregion_keyname = file_ttZ_signalregion_keyname[0].split('.')[:-1]
    file_ttZ_fakeable_keyname = input_path_ttZ_fakeable.split('/')[-1:]
    file_ttZ_fakeable_keyname = file_ttZ_fakeable_keyname[0].split('.')[:-1]

    file_ttWW_signalregion_keyname = input_path_ttWW_signalregion.split('/')[-1:]
    file_ttWW_signalregion_keyname = file_ttWW_signalregion_keyname[0].split('.')[:-1]

    file_EWK_signalregion_keyname = input_path_EWK_signalregion.split('/')[-1:]
    file_EWK_signalregion_keyname = file_EWK_signalregion_keyname[0].split('.')[:-1]

    file_data_signalregion_keyname = input_path_data_signalregion.split('/')[-1:]
    file_data_signalregion_keyname = file_data_signalregion_keyname[0].split('.')[:-1]

    for branch_index in xrange(len(branch_list)):
        branch_to_draw = []
        branch_to_draw.append(branch_list[branch_index])
        files_histname_dict = CPlotter.load_histos(files_, branch_to_draw,'syncTree')

        # Finish building histogram names (check 'load_histograms' in plotting class).
        file_ttH_HWW_Jet_numLoose_key = file_ttH_HWW_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"
        file_ttH_HZZ_Jet_numLoose_key = file_ttH_HZZ_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"
        file_ttH_Htautau_Jet_numLoose_key = file_ttH_Htautau_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"
        file_ttH_Hmm_Jet_numLoose_key = file_ttH_Hmm_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"
        file_ttH_Hoth_Jet_numLoose_key = file_ttH_Hoth_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"
        file_ttH_training_Jet_numLoose_key = file_ttH_training_keyname[0] + '_' + branch_list[branch_index] + '_loose_TrainingRegion'
        file_ttH_fakeable_Jet_numLoose_key = file_ttH_fakeable_keyname[0] + '_' + branch_list[branch_index] + '_fakeable_TrainingRegion'

        file_Conv_Jet_numLoose_key = file_Conv_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"
        file_Fakes_Jet_numLoose_key = file_Fakes_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"
        file_Flips_Jet_numLoose_key = file_Flips_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"
        file_ttJ_training_Jet_numLoose_key = file_ttJ_training_keyname[0] + '_' + branch_list[branch_index] + '_loose_TrainingRegion'
        file_ttJ_fakeable_Jet_numLoose_key = file_ttJ_fakeable_keyname[0] + '_' + branch_list[branch_index] + '_fakeable_TrainingRegion'

        file_ttW_signalregion_Jet_numLoose_key = file_ttW_signalregion_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"
        file_ttW_training_Jet_numLoose_key = file_ttW_training_keyname[0] + '_' + branch_list[branch_index] + '_loose_TrainingRegion'
        file_ttW_fakeable_Jet_numLoose_key = file_ttW_fakeable_keyname[0] + '_' + branch_list[branch_index] + '_fakeable_TrainingRegion'

        file_ttZ_signalregion_Jet_numLoose_key = file_ttZ_signalregion_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"
        file_ttZ_training_Jet_numLoose_key = file_ttZ_training_keyname[0] + '_' + branch_list[branch_index] + '_loose_TrainingRegion'
        file_ttZ_fakeable_Jet_numLoose_key = file_ttZ_fakeable_keyname[0] + '_' + branch_list[branch_index] + '_fakeable_TrainingRegion'

        file_data_signalregion_Jet_numLoose_key = file_data_signalregion_keyname[0] + '_' + branch_list[branch_index] + "_SignalRegion"


        # Get loose training region hisograms
        ttH_training_region_hist = files_histname_dict.get(file_ttH_training_Jet_numLoose_key)
        ttJ_training_region_hist = files_histname_dict.get(file_ttJ_training_Jet_numLoose_key)
        ttW_training_region_hist = files_histname_dict.get(file_ttW_training_Jet_numLoose_key)
        ttZ_training_region_hist = files_histname_dict.get(file_ttZ_training_Jet_numLoose_key)

        # Get fakeable training region histograms
        ttH_fakeable_training_region_hist = files_histname_dict.get(file_ttH_fakeable_Jet_numLoose_key)
        ttJ_fakeable_training_region_hist = files_histname_dict.get(file_ttJ_fakeable_Jet_numLoose_key)
        ttW_fakeable_training_region_hist = files_histname_dict.get(file_ttW_fakeable_Jet_numLoose_key)
        ttZ_fakeable_training_region_hist = files_histname_dict.get(file_ttZ_fakeable_Jet_numLoose_key)

        #Get all signal region histograms that are associated with trainign region processes
        hist_ttH_HWW = files_histname_dict.get(file_ttH_HWW_Jet_numLoose_key)
        hist_ttH_HZZ = files_histname_dict.get(file_ttH_HZZ_Jet_numLoose_key)
        hist_ttH_Htautau = files_histname_dict.get(file_ttH_Htautau_Jet_numLoose_key)
        hist_ttH_Hmm = files_histname_dict.get(file_ttH_Hmm_Jet_numLoose_key)
        hist_ttH_Hoth = files_histname_dict.get(file_ttH_Hoth_Jet_numLoose_key)
        hist_fakes = files_histname_dict.get(file_Fakes_Jet_numLoose_key)
        hist_flips = files_histname_dict.get(file_Flips_Jet_numLoose_key)
        hist_convs = files_histname_dict.get(file_Conv_Jet_numLoose_key)
        hist_ttW_SR = files_histname_dict.get(file_ttW_signalregion_Jet_numLoose_key)
        hist_ttZ_SR = files_histname_dict.get(file_ttZ_signalregion_Jet_numLoose_key)

        hist_data_SR = files_histname_dict.get(file_data_signalregion_Jet_numLoose_key)


        # Combine processes from signal region
        '''combined_hist_ttH_SR_title = branch_list[branch_index]+'_combined_ttH'
        combined_hist_ttH_SR = CPlotter.sum_hists([hist_ttH_HWW,hist_ttH_HZZ,hist_ttH_Htautau,hist_ttH_Hmm,hist_ttH_Hoth], combined_hist_ttH_SR_title)
        combined_hist_ttJ_SR_title = branch_list[branch_index]+'_combined_ttJ'
        combined_hist_ttJ_SR = CPlotter.sum_hists([hist_fakes,hist_flips,hist_convs], combined_hist_ttJ_SR_title)
        combined_hist_ttW_SR_title = branch_list[branch_index]+'_ttW'
        combined_hist_ttW_SR = CPlotter.sum_hists([hist_ttW_SR], combined_hist_ttW_SR_title)
        combined_hist_ttZ_SR_title = branch_list[branch_index]+'_ttZ'
        combined_hist_ttZ_SR = CPlotter.sum_hists([hist_ttZ_SR], combined_hist_ttZ_SR_title)'''

        combined_hist_MC_SR_title = branch_list[branch_index]+'_combined_MC'
        #combined_hist_MC_SR = CPlotter.sum_hists([hist_ttH_HWW,hist_ttH_HZZ,hist_ttH_Htautau,hist_ttH_Hmm,hist_ttH_Hoth,hist_fakes,hist_flips,hist_convs,hist_ttW_SR,hist_ttZ_SR], combined_hist_MC_SR_title)
        #combined_hist_MC_SR = CPlotter.stack_hists([hist_ttH_HWW,hist_ttH_HZZ,hist_ttH_Htautau,hist_ttH_Hmm,hist_ttH_Hoth,hist_fakes,hist_flips,hist_convs,hist_ttW_SR,hist_ttZ_SR], combined_hist_MC_SR_title)
        stacked_hist_MC_SR = CPlotter.stack_hists(files_histname_dict, combined_hist_MC_SR_title, branch_list[branch_index])


        # Make the histogram comparison plots
        '''CPlotter.make_hist(combined_hist_ttH_SR, ttH_training_region_hist, ttH_fakeable_training_region_hist, 'ttH', branch_list[branch_index])
        CPlotter.make_hist(combined_hist_ttJ_SR, ttJ_training_region_hist, ttJ_fakeable_training_region_hist, 'ttJ', branch_list[branch_index])
        CPlotter.make_hist(combined_hist_ttW_SR, ttW_training_region_hist, ttW_fakeable_training_region_hist, 'ttW', branch_list[branch_index])
        CPlotter.make_hist(combined_hist_ttZ_SR, ttZ_training_region_hist, ttZ_fakeable_training_region_hist, 'ttZ', branch_list[branch_index])'''

        #CPlotter.make_dataMC_comparison(stacked_hist_MC_SR, hist_data_SR, branch_list[branch_index])

    exit(0)

main()
