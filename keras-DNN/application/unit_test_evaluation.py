from evaluation.apply_DNN import apply_DNN
import matplotlib.pyplot as plt
import numpy as np
import numpy
import pandas
import pandas as pd
import optparse, json, argparse
import ROOT
import sys
from array import array
sys.path.insert(0, '/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/keras-DNN/')
from plotting.plotter import plotter
from ROOT import TFile, TTree, gDirectory, gPad
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
import os
from os import environ

def main():
    print ''
    DNN_applier = apply_DNN()

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)

    parser.add_argument('-p', '--processName', dest='processName', help='Process name. List of options in keys of process_filename dictionary', default=[], type=str, nargs='+')
    parser.add_argument('-r', '--region', dest='region', help='Option to choose e.g. DiLepRegion', default='DiLepRegion', type=str)
    parser.add_argument('-j', '--JES', dest='JES', help='Option to choose whether to run on JES Syst samples (0=Nominal, 1=JESUp, 2=JESDown)', default=0, type=int)
    parser.add_argument('-s', '--sel', dest='selection', help='Option to choose selection', default='tH', type=str)
    parser.add_argument('-y', '--year', dest='year', help='Option to choose year settings', default='2017', type=str)

    args = parser.parse_args()
    processes = args.processName
    region = args.region
    JES_flag = args.JES
    selection = args.selection
    nClasses = 4
    annum=args.year

    print '<unit_test_evaluation> Succesfully parsed arguments: processName= [%s], region= %s, JES_flag= %s , selection= %s' %(processes, region, JES_flag, selection)

    #outputname = '2017samples_tH_tunedweights_%s' % (selection)
    outputname = 'debug_%s' % (selection)

    input_var_jsonFile = ''

    if JES_flag==1:
        outputname = outputname+'_JESUp'
    if JES_flag==2:
        outputname = outputname+'_JESDown'

    # Open and load input variable .json

    input_var_jsonFile = open('../input_vars_SigRegion_wFwdJet.json','r')
    variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()

    # Append variables to a list of column headers for .csv file later
    column_headers = []
    for key,var in variable_list:
        column_headers.append(key)
    column_headers.append('EventWeight')
    column_headers.append('nEvent')

    if JES_flag == 0:
        JESname = ''
    elif JES_flag == 1:
        JESname = 'JESUp'
    elif JES_flag == 2:
        JESname = 'JESDown'

    # Dictionary of filenames to be run over along with their keys.
    process_filename = {
    'ttH_HWW' : ('TTH_hww_'+JESname+region),
    'ttH_Hmm' : ('TTH_hmm_'+JESname+region),
    'ttH_Htautau' : ('TTH_htt_'+JESname+region),
    'ttH_HZZ' : ('TTH_hzz_'+JESname+region),
    'ttH_HZG' : ('TTH_hzg_'+JESname+region),
    'tHq_HWW' : ('THQ_hww_'+JESname+region),
    'tHq_Htautau' : ('THQ_htt_'+JESname+region),
    'tHq_HZZ' : ('THQ_hzz_'+JESname+region),
    'tHq_HMM' : ('THQ_hmm_'+JESname+region),
    'tHq_HZG' : ('THQ_hzg_'+JESname+region),
    'tHW_HWW' : ('THW_hww_'+JESname+region),
    'tHW_Htautau' : ('THW_htt_'+JESname+region),
    'tHW_HZZ' : ('THW_hzz_'+JESname+region),
    'tHW_HMM' : ('THW_hmm_'+JESname+region),
    'tHW_HZG' : ('THW_hzg_'+JESname+region),
    'ttWW' : ('TTWW_'+JESname+region),
    'ttW' : ('TTW_'+JESname+region),
    'ttZ' : ('TTZ_'+JESname+region),
    'Conv' : ('Convs_'+JESname+region),
    'EWK' : ('EWK_'+JESname+region),
    'Fakes' : ('Fakes_'+JESname+region),
    'Flips' : ('Flips_'+JESname+region),
    'Rares' : ('Rares_'+JESname+region),
    'FakeSub' : ('FakeSub_'+JESname+region),
    'ttbar_closure' : ('TT_Clos'+JESname+region),
    'Data' : ('Data_'+JESname+region)
    }

    # Remove 'nEvent' from columns that will be used during in training
    training_columns = column_headers[:-2]
    num_variables = len(training_columns)

    # Name of directory that contains trained MVA model to apply.
    input_models_path = ''

    if selection == 'tH':
        input_models_path = ['2017samples_tH_tunedweights']

    # Load trained model
    optimizer = 'Adam'
    model_name_1 = os.path.join('../',input_models_path[0],'model.h5')
    model_1 = DNN_applier.load_trained_model(model_name_1, num_variables, optimizer, nClasses)

    # Make instance of plotter class
    Plotter = plotter()

    # Lists for all events in all files. Used to make diagnostic plots of networks performance over all samples.
    true_process = []
    model1_probs_ = []
    model1_pred_process = []
    EventWeights_ = []

    # Now loop over all samples
    for process in processes:
        print '<unit_test_evaluation> Process: ', process
        current_sample_name = process_filename.get(process)

        # Use JES flag to decide if we are running on a JES varied sample or not.
        if JES_flag==1:
            inputs_file_path = 'b/binghuan/Rootplas/Legacy/rootplas_LegacyAll_1110/%s%s/' % ('JESUp',region)
        elif JES_flag==2:
            inputs_file_path = 'b/binghuan/Rootplas/Legacy/rootplas_LegacyAll_1110/%s%s/' % ('JESDown',region)
        else:
            inputs_file_path = 'b/binghuan/Rootplas/Legacy/rootplas_LegacyAll_1110/%s/%s/%s/' % (region,annum,region)

        print '<unit_test_evaluation> Input file directory: ', inputs_file_path

        # Make final output directory
        samples_dir_w_appended_DNN = 'samples_w_DNN'
        if not os.path.exists(samples_dir_w_appended_DNN):
            os.makedirs(samples_dir_w_appended_DNN)
        samples_final_path_dir = os.path.join(samples_dir_w_appended_DNN,outputname)
        if not os.path.exists(samples_final_path_dir):
            os.makedirs(samples_final_path_dir)

        if JES_flag == 1:
            JES_label = 'JESUp'
        elif JES_flag == 2:
            JES_label = 'JESDown'
        else:
            JES_label = 'nominal'

        dataframe_name = '%s/%s_dataframe_%s_%s.csv' %(samples_final_path_dir,process,region,JES_label)
        if os.path.isfile(dataframe_name):
            print '<unit_test_evaluation> Loading %s . . . . ' % dataframe_name
            data = pandas.read_csv(dataframe_name)
        else:
            print '<unit_test_evaluation> Making *new* data file from %s . . . . ' % (inputs_file_path)
            print '<unit_test_evaluation> Applying selection ', selection
            selection_criteria=''
            if selection=='tH':
                selection_criteria = '(is_tH_like_and_not_ttH_like==0 || is_tH_like_and_not_ttH_like==1)'#&& n_presel_jet>=3'
            data = DNN_applier.load_data(inputs_file_path,column_headers,selection_criteria,process,process_filename.get(process))
            if len(data) == 0 :
                print '<unit_test_evaluation> No data! Next file.'
                continue
            print 'Saving new data .csv file at %s . . . . ' % (dataframe_name)
            data.to_csv(dataframe_name, index=False)

        nEvent = data['nEvent']

        # Using dataset like this instead of for loop over entries for predictions (necessary for keras).
        print '<unit_test_evaluation> Input features: ', training_columns
        X_test = data.iloc[:,0:num_variables]
        X_test = X_test.values
        result_probs_ = model_1.predict(np.array(X_test))

        # Create dictionary where the value is the array of probabilities for the four categories and the key is the event number.
        eventnum_resultsprob_dict = {}
        for index in range(result_probs_.shape[0]):
            eventnum_resultsprob_dict[nEvent[index]] = result_probs_[index]
            model1_probs_.append(result_probs_[index])


        inputlist = DNN_applier.getEOSlslist(directory=inputs_file_path+current_sample_name+".root")
        current_file = str(inputlist[0])
        print '<unit_test_evaluation> Input file: ', current_file

        # Open files and load ttrees
        data_file = TFile.Open(current_file)
        data_tree = data_file.Get("syncTree")

        # Check if input file is zombie
        if data_file.IsZombie():
            raise IOError('missing file')

        output_file_name = '%s/%s.root' % (samples_final_path_dir,process_filename.get(process))
        print '<unit_test_evaluation> Creating new output .root file'
        output_file = TFile.Open(output_file_name,'RECREATE')

        # CloneTree(nentries) - here copying none of the actually entries
        output_tree = data_tree.CloneTree(0)
        output_tree.SetName("output_tree")

        # Turn off all branches except ones you need if you want to speed up run time?
        output_tree.SetBranchStatus('*',1)

        # Append DNN Branches to new TTree
        # Add branches for values from highest output node and sentinel values for other nodes i.e. 'categorised'
        eval_ttHnode_cat = array('f',[0.])
        eval_Othernode_cat = array('f',[0.])
        eval_ttWnode_cat = array('f',[0.])
        eval_tHQnode_cat = array('f',[0.])
        ttH_branch_cat = output_tree.Branch('DNN_ttHnode_cat', eval_ttHnode_cat, 'DNN_ttHnode_cat/F')
        Other_branch_cat = output_tree.Branch('DNN_Othernode_cat', eval_Othernode_cat, 'DNN_Othernode_cat/F')
        ttW_branch_cat = output_tree.Branch('DNN_ttWnode_cat', eval_ttWnode_cat, 'DNN_ttWnode_cat/F')
        tHQ_branch_cat = output_tree.Branch('DNN_tHQnode_cat', eval_tHQnode_cat, 'DNN_tHQnode_cat/F')

        # un-categorised DNN variables
        eval_ttHnode_all = array('f',[0.])
        eval_Othernode_all = array('f',[0.])
        eval_ttWnode_all = array('f',[0.])
        eval_tHQnode_all = array('f',[0.])
        ttH_branch_all = output_tree.Branch('DNN_ttHnode_all', eval_ttHnode_all, 'DNN_ttHnode_all/F')
        Other_branch_all = output_tree.Branch('DNN_othernode_all', eval_Othernode_all, 'DNN_Othernode_all/F')
        ttW_branch_all = output_tree.Branch('DNN_ttWnode_all', eval_ttWnode_all, 'DNN_ttWnode_all/F')
        tHQ_branch_all = output_tree.Branch('DNN_tHQnode_all', eval_tHQnode_all, 'DNN_tHQnode_all/F')

        # Now add branches conatining the max value for each event and the category for each event
        eval_maxval = array('f',[0.])
        DNNCat = array('f',[0.])
        DNNmaxval_branch = output_tree.Branch('DNN_maxval', eval_maxval, 'DNN_maxval/F')
        DNNCat_branch = output_tree.Branch('DNNCat', DNNCat, 'DNNCat/F')

        sample_name = process
        histoname_type = 'Category'

        histo_ttHclassified_events_title = 'ttH %s Events: %s Sample' % (histoname_type,sample_name)
        histo_ttHclassified_events_name = 'histo_ttH%s_events_%s' % (histoname_type,sample_name)
        histo_ttHclassified_events = ROOT.TH1D(histo_ttHclassified_events_name,histo_ttHclassified_events_title,200,0,1.)
        histo_Otherclassified_events_title = 'Other %s Events: %s Sample' % (histoname_type,sample_name)
        histo_Otherclassified_events_name = 'histo_Other%s_events_%s' % (histoname_type,sample_name)
        histo_Otherclassified_events = ROOT.TH1D(histo_Otherclassified_events_name,histo_Otherclassified_events_title,200,0,1.)
        histo_ttWclassified_events_title = 'ttW %s Events: %s Sample' % (histoname_type,sample_name)
        histo_ttWclassified_events_name = 'histo_ttW%s_events_%s' % (histoname_type,sample_name)
        histo_ttWclassified_events = ROOT.TH1D(histo_ttWclassified_events_name,histo_ttWclassified_events_title,200,0,1.)
        histo_tHQclassified_events_title = 'tHQ %s Events: %s Sample' % (histoname_type,sample_name)
        histo_tHQclassified_events_name = 'histo_tHQ%s_events_%s' % (histoname_type,sample_name)
        histo_tHQclassified_events = ROOT.TH1D(histo_tHQclassified_events_name,histo_tHQclassified_events_title,200,0,1.)

        temp_percentage_done = 0
        uniqueEventID = []

        ######## Loop over ttree #########

        print '<unit_test_evaluation> data_tree # Entries: ', data_tree.GetEntries()
        if output_tree.GetEntries() != 0:
            print '<unit_test_evaluation> output_tree # Entries: ', output_tree.GetEntries()
            print 'This tree should be empty at this point!!!!! check cloning correctly'

        for i in range(data_tree.GetEntries()):
            eval_ttHnode_cat[0]= -1.
            eval_Othernode_cat[0]= -1.
            eval_ttWnode_cat[0]= -1.
            eval_tHQnode_cat[0]= -1.
            eval_ttHnode_all[0]= -1.
            eval_Othernode_all[0]= -1.
            eval_ttWnode_all[0]= -1.
            eval_tHQnode_all[0]= -1.
            eval_maxval[0]= -1.
            DNNCat[0] = -1.

            percentage_done = int(100*float(i)/float(data_tree.GetEntries()))
            if percentage_done % 10 == 0:
                if percentage_done != temp_percentage_done:
                    print percentage_done
                    temp_percentage_done = percentage_done
            data_tree.GetEntry(i)

            Eventnum_ = array('d',[0])
            Eventnum_ = data_tree.nEvent
            EventWeight_ = array('d',[0])
            EventWeight_ = data_tree.EventWeight
            xsec_rwgt_ = array('d',[0])
            xsec_rwgt_ = data_tree.xsec_rwgt

            n_presel_jet  = array('d',[0])
            n_presel_jet = data_tree.n_presel_jet
            is_tH_like_and_not_ttH_like = array('d',[0])
            is_tH_like_and_not_ttH_like = output_tree.is_tH_like_and_not_ttH_like

            if (is_tH_like_and_not_ttH_like == 0 or is_tH_like_and_not_ttH_like == 1): #and n_presel_jet>=3:
                pass_selection = 1
            else:
                pass_selection = 0

            if selection == 'tH':
                if pass_selection==0:
                    continue
            else:
                print 'NO selection applied!'



            '''if Eventnum_ in uniqueEventID:
                print 'Eventnum_ : %s already exists ' % Eventnum_
                continue
            else:
                uniqueEventID.append(Eventnum_)
            '''


            if 'ttH_' in process:
                true_process.append(0)
            elif 'Fakes' in process or 'Flips' in process:
                true_process.append(1)
            elif 'ttW' in process:
                true_process.append(2)
            elif 'tHq' in process:
                true_process.append(3)
            else:
                true_process.append(4)

            EventWeights_.append(EventWeight_)

            evaluated_node_values = []
            #print 'Eventnum_: ', Eventnum_
            #for key,var in variable_list:
            #    print 'key: %s, value: %s' % (key , data_tree.GetLeaf(key).GetValue())
            #print 'True process: ', true_process
            # Get the value for event on each of the DNN nodes
            evaluated_node_values = DNN_applier.evaluate_model(eventnum_resultsprob_dict,Eventnum_)
            #print 'evaluated_node_values: ', evaluated_node_values
            # Get the maximum output value
            maxval = max(evaluated_node_values)
            # Find the max value in and return its position (i.e. node classification)
            event_classification = evaluated_node_values.index(maxval)
            #print 'event_classification: ', event_classification
            # Append classification value to list of predictions
            model1_pred_process.append(event_classification)
            #print 'model1_pred_process: ', model1_pred_process

            eval_ttHnode_all[0] = evaluated_node_values[0]
            eval_Othernode_all[0] = evaluated_node_values[1]
            eval_ttWnode_all[0] = evaluated_node_values[2]
            eval_tHQnode_all[0] = evaluated_node_values[3]

            DNNCat[0] = float(event_classification)
            eval_maxval[0] = evaluated_node_values[event_classification]

            if event_classification == 0:
                histo_ttHclassified_events.Fill(evaluated_node_values[0],EventWeight_)
                eval_ttHnode_cat[0] = evaluated_node_values[0]
                eval_Othernode_cat[0] = -1.
                eval_ttWnode_cat[0] = -1.
                eval_tHQnode_cat[0] = -1.
            elif event_classification == 1:
                histo_Otherclassified_events.Fill(evaluated_node_values[1],EventWeight_)
                eval_ttHnode_cat[0] = -1.
                eval_Othernode_cat[0] = evaluated_node_values[1]
                eval_ttWnode_cat[0] = -1.
                eval_tHQnode_cat[0] = -1.
            elif event_classification == 2:
                histo_ttWclassified_events.Fill(evaluated_node_values[2],EventWeight_)
                eval_ttHnode_cat[0] = -1.
                eval_Othernode_cat[0] = -1.
                eval_ttWnode_cat[0] = evaluated_node_values[2]
                eval_tHQnode_cat[0] = -1.
            elif event_classification == 3:
                histo_tHQclassified_events.Fill(evaluated_node_values[3],EventWeight_)
                eval_ttHnode_cat[0] = -1.
                eval_Othernode_cat[0] = -1.
                eval_ttWnode_cat[0] = -1.
                eval_tHQnode_cat[0] = evaluated_node_values[3]
            else:
                histo_ttHclassified_events.Fill(-1.,EventWeight_)
                histo_Otherclassified_events.Fill(-1.,EventWeight_)
                histo_ttWclassified_events.Fill(-1.,EventWeight_)
                histo_tHQclassified_events.Fill(-1.,EventWeight_)
                eval_ttHnode_cat[0] = -1.
                eval_Othernode_cat[0] = -1.
                eval_ttWnode_cat[0] = -1.
                eval_tHQnode_cat[0] = -1.
                print '<unit_test_evaluation> NO classification for event!?'
                continue

            output_tree.Fill()

        print '<unit_test_evaluation> Clear # event - DNN result dictionary'
        eventnum_resultsprob_dict.clear()
        print '<unit_test_evaluation> Write output file : %s ' % (output_file_name)
        output_file.Write()
        print '<unit_test_evaluation> Close output file'
        output_file.Close()
        print '<unit_test_evaluation> Close input file'
        data_file.Close()

    plots_dir = os.path.join(samples_final_path_dir,'plots/')
    Plotter.plots_directory = plots_dir

    Plotter.conf_matrix(true_process,model1_pred_process,EventWeights_,'')
    Plotter.save_plots(dir=plots_dir, filename='yields_non_norm_confusion_matrix_APPL.png')
    Plotter.conf_matrix(true_process,model1_pred_process,EventWeights_,'index')
    Plotter.save_plots(dir=plots_dir, filename='yields_norm_confusion_matrix_APPL.png')

    model1_probs_ = np.array(model1_probs_)
    Plotter.ROC_sklearn(true_process, model1_probs_, true_process, model1_probs_, 0 , 'ttHnode')
    Plotter.ROC_sklearn(true_process, model1_probs_, true_process, model1_probs_, 1 , 'Othernode')
    Plotter.ROC_sklearn(true_process, model1_probs_, true_process, model1_probs_, 2 , 'ttWnode')
    Plotter.ROC_sklearn(true_process, model1_probs_, true_process, model1_probs_, 3 , 'tHQnode')

    exit(0)

main()
