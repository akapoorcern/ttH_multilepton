import ROOT, os, optparse, math, json, sys
from ROOT import TFile, TString,TLegend, THStack, TLatex, TH1, TList, TGaxis
from array import array
from os.path import isfile

def GetDataOverMC(stack_mc, histo_data, plotname):

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
    DOverMC.GetXaxis().SetTitle(plotname)
    DOverMC.GetXaxis().SetLabelSize(0.15)
    DOverMC.GetXaxis().SetTitleSize(0.15)
    DOverMC.GetXaxis().SetTitleOffset(1.)
    DOverMC.SetFillStyle(0)
    DOverMC.SetMarkerStyle(2)
    DOverMC.SetMarkerColor(1)
    DOverMC.SetLineColor(1)
    return DOverMC

def category_selection(analysis_bin, data_tree):

    selection_flag_leptons = 0

    if 'ee' in analysis_bin:
        if 'neg' in analysis_bin:
            if (data_tree.SubCat2l == 1):
                selection_flag_leptons = 1
        elif 'pos' in analysis_bin:
            if (data_tree.SubCat2l == 2):
                selection_flag_leptons = 1
        else:
            if (data_tree.SubCat2l == 1) or (data_tree.SubCat2l == 2):
                selection_flag_leptons = 1
    elif 'em' in analysis_bin:
        if 'neg' in analysis_bin:
            if (data_tree.SubCat2l == 3) or (data_tree.SubCat2l == 5):
                selection_flag_leptons = 1
        elif 'pos' in analysis_bin:
            if (data_tree.SubCat2l == 4) or (data_tree.SubCat2l == 6):
                selection_flag_leptons = 1
        else:
            if (data_tree.SubCat2l == 3) or (data_tree.SubCat2l == 4) or (data_tree.SubCat2l == 5) or (data_tree.SubCat2l == 6):
                selection_flag_leptons = 1
    elif 'mm' in analysis_bin:
        if 'neg' in analysis_bin:
            if (data_tree.SubCat2l == 7) or (data_tree.SubCat2l == 9):
                selection_flag_leptons = 1
        elif 'pos' in analysis_bin:
            if (data_tree.SubCat2l == 8) or (data_tree.SubCat2l == 10):
                selection_flag_leptons = 1
        else:
            if (data_tree.SubCat2l == 7) or (data_tree.SubCat2l == 8) or (data_tree.SubCat2l == 9) or (data_tree.SubCat2l == 10):
                selection_flag_leptons = 1

    selection_flag_bjets = 0
    if 'bloose' in analysis_bin:
        if data_tree.nBJetMedium < 2:
            selection_flag_bjets = 1
    elif 'bmedium' in analysis_bin:
        if data_tree.nBJetMedium >= 2:
            selection_flag_bjets = 1
    else:
        selection_flag_bjets = 1

    return selection_flag_leptons * selection_flag_bjets

def main():
    usage = 'usage: %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('-i', '--input_directory', dest='input_directory', help='input directory with .root files to plot distributions from.')
    parser.add_option('-a', dest='all_flag', action="store_true", help='Flag = True if you need to create the histograms rootfile')
    parser.add_option('-m', dest='all_flag', action="store_false", help='Flag = False if you simply want to draw the histograms from the file')
    (opt,args) = parser.parse_args()


    all_flag = opt.all_flag
    print 'all_flag: ', all_flag

    #output_file_name = 'ControlPlots/DNN_input_variables.root'
    #output_file_name = 'ControlPlots/DNN_input_variables-HJ1-metLD-maxet-etc.root'
    #output_file_name = 'ControlPlots/DNN_input_variables_nJets.root'

    # Possible variable groups:
    # nJets
    # jet_kin
    # lep_kin
    # lep_charge
    # jet_taggers
    # jet_lep_kin
    # discs
    # other
    variable_group = 'lep_charge'
    channel_name = ['2L']
    #channel_name = ['ttWctrl']

    # Use full process list when running on nominal
    #process_list = ['TTH_hww','TTH_htt','TTH_hmm','TTH_hot','Conv','EWK','Fakes','Flips','Rares','TTW','TTZ','TTWW','Data']
    process_list = ['Conv','EWK','Fakes','Flips','Rares','TTW','TTZ','TTWW','TTH_hww','TTH_htt','TTH_hot','Data']

    #bin_list = ['ee_neg','ee_pos','em_neg_bloose','em_pos_bloose','em_neg_bmedium','em_pos_bmedium','mm_neg_bloose','mm_pos_bloose','mm_neg_bmedium','mm_pos_bmedium']
    bin_list = ['ee_neg','ee_pos']

    # DNN input variables - these are the variables that will be plotted whe the script runs.
    variables_list = []
    if variable_group == 'nJets':
        variables_list = ['nBJetLoose', 'nBJetMedium','Jet_numLoose']
    if variable_group == 'jet_kin':
        variables_list = ['jet1_pt','jet1_eta','jet1_phi','jet1_E','jet2_pt','jet2_eta','jet2_phi','jet2_E','jet3_pt','jet3_eta','jet3_phi','jet3_E']
    if variable_group == 'lep_kin':
        variables_list = ['lep1_conePt','lep1_eta','lep1_phi','lep1_E','lep2_conePt','lep2_eta','lep2_phi','lep2_E']
    if variable_group == 'lep_charge':
        variables_list = ['lep1_charge','lep2_charge']
    if variable_group == 'jet_taggers':
        variables_list = ['Hj1_BDT','hadTop_BDT']##Currently, restop_BDT is not in the signal region ntuples
    if variable_group == 'jet_lep_kin':
        variables_list = ['mindr_lep1_jet','mindr_lep2_jet','mindr_lep3_jet']
    if variable_group == 'discs':
        variables_list = ['Bin2l']
    if variable_group == 'other':
        variables_list = ['metLD','maxeta','PFMET','massL']

    # DO NOT DELETE !!!
    # Dictionary containing binning options for all histograms [number, xmin, xmax]
    # Contains binning options for histograms not plotted when script runs as they may have been
    # run in the past.
    # DO NOT DELETE !!! as we may want to run again in the future in which case we
    # just add to the variable list (which could be externalised to a .json).
    bin_dic = {
    "nLooseJet" : [10,0,10],
    "Jet_numLoose" : [10,0,10],
    "n_presel_jet" : [10,0,10],
    "nBJetLoose" : [7,0,7],
    "nBJetMedium" : [7,0,7],
    "jet1_pt" : [30,0,300],
    "jet1_eta" : [20,-3,3],
    "jet1_phi" : [20,-5,5],
    "jet1_E" : [30,0,300],
    "jet2_pt" : [30,0,250],
    "jet2_eta" : [20,-3,3],
    "jet2_phi" : [20,-5,5],
    "jet2_E" : [30,0,250],
    "jet3_pt" : [30,0,200],
    "jet3_eta" : [20,-3,3],
    "jet3_phi" : [20,-5,5],
    "jet3_E" : [30,0,200],
    "jet4_pt" : [30,0,150],
    "jet4_eta" : [20,-3,3],
    "jet4_phi" : [20,-5,5],
    "jet4_E" : [30,0,200],
    "lep1_conePt" : [30,0,250],
    "lep1_eta" : [20,-3,3],
    "lep1_phi" : [20,-5,5],
    "lep1_E" : [30,0,300],
    "lep1_charge" : [10,-5,5],
    "lep2_conePt" : [30,0,200],
    "lep2_eta" : [20,-3,3],
    "lep2_phi" :  [20,-5,5],
    "lep2_E" : [30,0,200],
    "lep2_charge" : [10,-5,5],
    "PFMET" : [30,0,250],
    "Hj_tagger" : [10,-2,1],
    "HTT" : [10,-1,2],
    "Bin2l" : [13,0,13],
    "resTop_BDT" : [10,-1,2],
    "hadTop_BDT" : [10,-1,2],
    "Hj1_BDT" : [10,-2,1],
    "metLD" : [50,0,300],
    "maxeta" : [10,0,2.5],
    "mindr_lep1_jet" : [10,0,3.5],
    "mindr_lep2_jet" : [10,0,3.5],
    "mindr_lep3_jet" : [10,0,3.5],
    "jet1_CSV" : [10,0,1],
    "jet2_CSV" : [10,0,1],
    "jet3_CSV" : [10,0,1],
    "jet4_CSV" : [10,0,1],
    "massL" : [20,0,300]
    }

    # Open input files and load ttrees:
    # Maybe want a flag here in case you don't want to make new hist files and simply rerun the second step creating the plots.
    for chan in channel_name:
        for analysis_bin in bin_list:
            output_file_name = 'ControlPlots/Dist_%s_%s_%s.root' %(variable_group,chan, analysis_bin)
            if all_flag == True:
                print 'create new histogram file'
                output_file = TFile.Open(output_file_name,'RECREATE')

            if opt.input_directory == None:
                print 'Input directory not defined! Please define option -i'
                sys.exit(1)

            input_directory_ = os.path.join(opt.input_directory,chan)
            print 'input_directory_: ', input_directory_

            var_index = 0
            print 'chan : ', chan
            #If all_flag is true, I need to create the histograms file as well as draw the histograms.
            if all_flag==True:
                for var in variables_list:
                    print 'variable : ', var
                    input_path = os.path.abspath(input_directory_)
                    binning_options = bin_dic[var]

                    for process_index in xrange(len(process_list)):
                        #Create one histogram per process per variable and bin very finely (can always rebin more broadly later).
                        histo_ = ROOT.TH1D("generic_name","generic_title",binning_options[0],binning_options[1],binning_options[2])
                        histo_.Sumw2()
                        process_name = process_list[process_index]
                        print 'process_index: ' , process_index
                        print 'process: ' , process_name
                        input_name_ = '%s/%s_%s.root' % (input_path, process_name, chan)
                        input_file_ = TFile.Open(input_name_)
                        if not os.path.isfile(input_name_):
                            print 'No such input file: %s' % input_name_
                        histo_name = 'histo_%s_%s_%s_%s' % (var, chan, process_name, analysis_bin)
                        histo_.SetName(histo_name)
                        histo_title = '%s %s Channel %s %s' % (var, chan, process_name, analysis_bin)
                        histo_.SetTitle(histo_title)
                        data_tree = input_file_.Get("syncTree")
                        branchlist = data_tree.GetListOfBranches()
                        nBranches = data_tree.GetNbranches()
                        print '# entries : ', data_tree.GetEntries()
                        for i in range(data_tree.GetEntries()):
                            data_tree.GetEntry(i)
                            isSelected = category_selection(analysis_bin, data_tree)
                            if isSelected == 0:
                                continue
                            variable = data_tree.GetBranch(var).GetLeaf(var)
                            weight = data_tree.GetBranch('EventWeight').GetLeaf('EventWeight')
                            if 'data' in process_name or 'Data' in process_name:
                                histo_.Fill(variable.GetValue(), 1.)
                            else:
                                histo_.Fill(variable.GetValue(), weight.GetValue())

                        output_file.cd()
                        histo_.Write()
                        histo_.SetDirectory(0)
                        histo_.Clear()
                    var_index = var_index + 1
                output_file.Close()
            # Ensure the histograms files exists

            if not os.path.isfile(output_file_name):
                print 'No such input file: %s' % output_file_name

            # output_file should have histograms for variables separated by process.
            process_colours = [5,6,13,17,38,8,3,9,2,2,2]
            for variable_index in xrange(0,len(variables_list)):
                output_file_ = TFile.Open(output_file_name,'READ')
                legend = TLegend(0.60,  0.70,  0.9,  0.9)
                legend.SetNColumns(2)
                hist_stack = ROOT.THStack()
                datahistname = 'histo_%s_%s_Data_%s' % (variables_list[variable_index], chan, analysis_bin)
                print 'datahistname = ', datahistname
                datahist = output_file_.Get(datahistname)
                datahist.Sumw2()

                index = 0
                for tkey in output_file_.GetListOfKeys():
                    key = tkey.GetName()
                    obj = output_file_.Get(key)
                    if not obj.InheritsFrom('TH1') : continue
                    if 'data' in key or 'Data' in key:
                        continue
                    if variables_list[variable_index] in key:
                        print 'tkey: ', tkey
                        for process_ in process_list:
                            if process_ in key:
                                print 'process name = ', process_
                        legend_label = "%s sample" % (process_list[index])
                        legend.AddEntry(obj,legend_label,'f')
                        obj.Sumw2()
                        obj.SetMarkerColor(process_colours[index])
                        obj.SetLineColor(process_colours[index])
                        obj.SetMarkerStyle(20)
                        obj.SetFillColor(process_colours[index])
                        obj.SetFillStyle(3001)
                        hist_stack.Add(obj)
                        index = index + 1

                hist_stack.ls()
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
                datahist.SetLineColor(1)
                datahist.SetMarkerColor(1)
                datahist.SetMarkerStyle(20)
                maxyaxis_stack = hist_stack.GetStack().Last().GetMaximum()
                maxyaxis_data = datahist.GetMaximum()
                maxyaxis = max(maxyaxis_stack, maxyaxis_data) + max(maxyaxis_stack, maxyaxis_data)/3
                hist_stack.SetMaximum(maxyaxis)
                hist_stack.Draw("HIST")
                datahist.Draw("HISTEPSAME")
                legend.Draw('same')

                txt2=ROOT.TLatex()
                txt2.SetNDC(True)
                txt2.SetTextFont(43)
                txt2.SetTextSize(18)
                txt2.SetTextAlign(12)
                txt2.DrawLatex(0.13,0.925,'#bf{CMS}')
                txt2.DrawLatex(0.2,0.92,'#it{Preliminary}')
                txt2.DrawLatex(0.57,0.925,'%3.1f fb^{-1} (13 TeV)' %(41860.080/1000.) )

                txt3=ROOT.TLatex()
                txt3.SetNDC(True)
                txt3.SetTextFont(43)
                txt3.SetTextSize(18)
                txt3.SetTextAlign(12)
                label = '#it{%s %s}' % (chan, analysis_bin)
                txt3.DrawLatex(0.13,0.825,label)

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

                ratiohist = GetDataOverMC(hist_stack,datahist,variables_list[variable_index])
                ratiohist.Draw("P")
                line = ROOT.TLine(ratiohist.GetXaxis().GetBinUpEdge(0),1,ratiohist.GetXaxis().GetBinUpEdge(ratiohist.GetNbinsX()),1);
                line.SetLineColor(2);
                line.Draw("same");
                c1.Update()

                test_output_histname_png = 'ControlPlots/control_hist_%s_%s_%s.png' % (variables_list[variable_index], chan, analysis_bin)
                c1.Print(test_output_histname_png,'png')

                test_output_histname_pdf = 'ControlPlots/control_hist_%s_%s_%s.pdf' % (variables_list[variable_index], chan, analysis_bin)
                c1.Print(test_output_histname_pdf,'pdf')

main()
