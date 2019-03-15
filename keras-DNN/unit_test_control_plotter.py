from plotting.control_plotter import control_plotter

def main():
    CPlotter = control_plotter()
    CPlotter.check_dir('invar_control_plots')
    files_list = []
    input_path_1 = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/rootplas_20190227/SigRegion/SigRegion/TTH_hww_SigRegion.root')
    input_path_2 = CPlotter.getEOSlslist(directory='/b/binghuan/Rootplas/TrainMVA_20190220/NoJetNCut/ttHnobb_NoJetNCut.root')
    files_list.append(input_path_1[0])
    files_list.append(input_path_2[0])
    files_ = CPlotter.open_files(files_list)
    branch_list = ['Jet_numLoose']
    input_files_branchname_dict = CPlotter.load_histos(files_,branch_list,'syncTree')
    CPlotter.make_hist(input_files_branchname_dict)

    exit(0)

main()
