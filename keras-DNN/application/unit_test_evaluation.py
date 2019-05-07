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
from ROOT import TFile, TTree, gDirectory
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

    number_of_classes = 4

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)

    parser.add_argument('-p', '--processName', dest='processName', help='Process name. List of options in keys of process_filename dictionary', default=[], type=str, nargs='+')
    parser.add_argument('-r', '--region', dest='region', help='Option to choose SigRegion or CtrlRegion', default='SigRegion', type=str)
    parser.add_argument('-j', '--JES', dest='JES', help='Option to choose whether to run on JES Syst samples (0=Nominal, 1=JESUp, 2=JESDown)', default=0, type=int)
    parser.add_argument('-s', '--sel', dest='selection', help='Option to choose selection', default='geq4j', type=str)

    args = parser.parse_args()
    processes = args.processName
    region = args.region
    JES_flag = args.JES
    selection = args.selection

    print 'region = ', region
    print 'JES_flag = ', JES_flag

    outputname = '2019_04_26_%s' % (selection)
    #outputname = 'test'

    input_var_jsonFile = ''
    if region == 'CtrlRegion':
        outputname = outputname + '_ttWctrl'
        if JES_flag==1:
            outputname = outputname+'_JESUp'
        if JES_flag==2:
            outputname = outputname+'_JESDown'
        region = 'ttWctrl'
        input_var_jsonFile = open('../input_vars_CtrlRegion.json','r')
    elif region == 'SigRegion':
        if JES_flag==1:
            outputname = outputname+'_JESUp'
        if JES_flag==2:
            outputname = outputname+'_JESDown'
        input_var_jsonFile = open('../input_vars_SigRegion.json','r')
    elif region == 'DiLepRegion':
        if JES_flag==1:
            outputname = outputname+'_JESUp'
        if JES_flag==2:
            outputname = outputname+'_JESDown'
        input_var_jsonFile = open('../input_vars_SigRegion.json','r')


    variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()
    column_headers = []
    for key,var in variable_list:
        if 'hadTop_BDT' in key:
            key = 'hadTop_BDT'
        if 'Hj1_BDT' in key:
            key = 'Hj1_BDT'
        if 'Hj_tagger_hadTop' in key:
            key = 'Hj_tagger_hadTop'
        column_headers.append(key)
    if region == 'ttWctrl':
        column_headers.append('Jet_numLoose')
    column_headers.append('nEvent')

    if JES_flag == 0:
        JESname = ''
    elif JES_flag == 1:
        JESname = 'JESUp'
    elif JES_flag == 2:
        JESname = 'JESDown'

    process_filename = {
    'ttH_HWW' : ('TTH_hww_'+JESname+region),
    'ttH_Hmm' : ('TTH_hmm_'+JESname+region),
    'ttH_Htautau' : ('TTH_htt_'+JESname+region),
    'ttH_HZZ' : ('TTH_hzz_'+JESname+region),
    'ttH_other' : ('TTH_hot_'+JESname+region),
    'tHq_HWW' : ('THQ_hww_'+JESname+region),
    'tHq_Htautau' : ('THQ_htt_'+JESname+region),
    'tHq_HZZ' : ('THQ_hzz_'+JESname+region),
    'tHW_HWW' : ('THW_hww_'+JESname+region),
    'tHW_Htautau' : ('THW_htt_'+JESname+region),
    'tHW_HZZ' : ('THW_hzz_'+JESname+region),
    'ttWW' : ('TTWW_'+JESname+region),
    'ttW' : ('TTW_'+JESname+region),
    'ttZ' : ('TTZ_'+JESname+region),
    'Conv' : ('Conv_'+JESname+region),
    'EWK' : ('EWK_'+JESname+region),
    'Fakes' : ('Fakes_'+JESname+region),
    'Flips' : ('Flips_'+JESname+region),
    'Rares' : ('Rares_'+JESname+region),
    'FakeSub' : ('FakeSub_'+JESname+region),
    'ttbar_closure' : ('TT_Clos'+JESname+region),
    'Data' : ('Data_'+JESname+region)
    }

    print 'column_headers: ', column_headers
    if region == 'ttWctrl':
        training_columns = column_headers[:-2]
    else:
        training_columns = column_headers[:-1]

    num_variables = len(training_columns)
    optimizer = 'Adam'

    print 'region = ', region
    input_models_path = ''
    # Want to be able to run on
    if region == 'SigRegion':
        input_models_path = ['2019-04-26_geq4j_InverseSRYields_SigRegion']
    elif region == 'ttWctrl':
        input_models_path = ['2019-04-26_eeq3j_InverseSRYields_CtrlRegion']
    elif region == 'DiLepRegion':
        input_models_path = ['2019-04-26_geq3j_InverseSRYields_SigRegion']

    model_name_1 = os.path.join('../',input_models_path[0],'model.h5')
    model_1 = DNN_applier.load_trained_model(model_name_1, num_variables, optimizer)


    Plotter = plotter()

    true_process = []
    model1_pred_process = []
    model2_pred_process = []
    model3_pred_process = []
    EventWeights_ = []

    for process in processes:
        print 'process: ', process
        current_sample_name = process_filename.get(process)
        if JES_flag==1:
            inputs_file_path = '/b/binghuan/Rootplas/rootplas_20190227/%s/%s%s/' % (region,'JESUp',region)
        elif JES_flag==2:
            inputs_file_path = '/b/binghuan/Rootplas/rootplas_20190227/%s/%s%s/' % (region,'JESDown',region)
        else:
            inputs_file_path = '/b/binghuan/Rootplas/rootplas_20190227/%s/%s/' % (region,region)

        print 'inputs_file_path = ', inputs_file_path

        samples_dir_w_appended_DNN = 'samples_w_DNN'
        if not os.path.exists(samples_dir_w_appended_DNN):
            os.makedirs(samples_dir_w_appended_DNN)

        samples_final_path_dir = os.path.join(samples_dir_w_appended_DNN,outputname)
        if not os.path.exists(samples_final_path_dir):
            os.makedirs(samples_final_path_dir)

        print samples_final_path_dir
        if JES_flag == 1:
            JES_label = 'JESUp'
        elif JES_flag == 2:
            JES_label = 'JESDown'
        else:
            JES_label = 'nominal'

        dataframe_name = '%s/%s_dataframe_%s_%s.csv' %(samples_final_path_dir,process,region,JES_label)
        if os.path.isfile(dataframe_name):
            print 'Loading %s . . . . ' % dataframe_name
            data = pandas.read_csv(dataframe_name)
        else:
            print 'Loading new data file from %s . . . . ' % (inputs_file_path)
            if 'SigRegion' in region or selection == 'geq4j':
                data = DNN_applier.load_data(inputs_file_path,column_headers,'Jet_numLoose>=4',process,process_filename.get(process))
            elif 'DiLepRegion' in region or selection == 'geq3j':
                data = DNN_applier.load_data(inputs_file_path,column_headers,'Jet_numLoose>=3',process,process_filename.get(process))
            elif region == 'ttWctrl' or selection == 'eeq3j':
                data = DNN_applier.load_data(inputs_file_path,column_headers,'Jet_numLoose==3',process,process_filename.get(process))
            if len(data) == 0 :
                print 'No data! Next file.'
                continue
            print 'Saving new data .csv file at %s . . . . ' % (dataframe_name)
            data.to_csv(dataframe_name, index=False)


        #Evweights = data['EventWeight']
        #xsec_rwgt = data['xsec_rwgt']
        nEvent = data['nEvent']

        print 'Use columns = ', training_columns
        print 'num_variables = ', num_variables
        #print data
        X_test = data.iloc[:,0:num_variables]
        X_test = X_test.values

        #scaler = StandardScaler(with_mean=False, with_std=False).fit(X_test)
        #scaler = StandardScaler().fit(X_test)
        #scaler = Normalizer('l2').fit(X_test)
        #scaler = Normalizer('l1').fit(X_test)
        #X_test = scaler.transform(X_test)

        result_probs_test = model_1.predict_proba(np.array(X_test))

        # create dictionary where the value is the array of probabilities for the four categories
        # and the key is the event number.
        eventnum_resultsprob_dict = {}
        for index in range(result_probs_test.shape[0]):
            eventnum_resultsprob_dict[nEvent[index]] = result_probs_test[index]


        inputlist = DNN_applier.getEOSlslist(directory=inputs_file_path+current_sample_name+".root")
        current_file = str(inputlist[0])

        # Open files and load ttrees
        data_file = TFile.Open(current_file)
        data_tree = data_file.Get("syncTree")

        output_file_name = '%s/%s.root' % (samples_final_path_dir,process_filename.get(process))
        output_file = TFile.Open(output_file_name,'RECREATE')
        output_tree = data_tree.CopyTree("")
        output_tree.SetName("output_tree")
        nEvents_check = output_tree.BuildIndex("nEvent","run")

        # DNN variables
        eval_ttHnode_all = array('f',[0.])
        eval_ttJnode_all = array('f',[0.])
        eval_ttWnode_all = array('f',[0.])
        eval_ttZnode_all = array('f',[0.])
        eval_ttHnode_cat = array('f',[0.])
        eval_ttJnode_cat = array('f',[0.])
        eval_ttWnode_cat = array('f',[0.])
        eval_ttZnode_cat = array('f',[0.])
        # DNN Branches
        ttH_branch_cat = output_tree.Branch('DNN_ttHnode_cat', eval_ttHnode_cat, 'DNN_ttHnode_cat/F')
        ttJ_branch_cat = output_tree.Branch('DNN_ttJnode_cat', eval_ttJnode_cat, 'DNN_ttJnode_cat/F')
        ttW_branch_cat = output_tree.Branch('DNN_ttWnode_cat', eval_ttWnode_cat, 'DNN_ttWnode_cat/F')
        ttZ_branch_cat = output_tree.Branch('DNN_ttZnode_cat', eval_ttZnode_cat, 'DNN_ttZnode_cat/F')
        ttH_branch_all = output_tree.Branch('DNN_ttHnode_all', eval_ttHnode_all, 'DNN_ttHnode_all/F')
        ttJ_branch_all = output_tree.Branch('DNN_ttJnode_all', eval_ttJnode_all, 'DNN_ttJnode_all/F')
        ttW_branch_all = output_tree.Branch('DNN_ttWnode_all', eval_ttWnode_all, 'DNN_ttWnode_all/F')
        ttZ_branch_all = output_tree.Branch('DNN_ttZnode_all', eval_ttZnode_all, 'DNN_ttZnode_all/F')

        eval_maxval = array('f',[0.])
        DNNCat = array('f',[0.])
        DNNmaxval_branch = output_tree.Branch('DNN_maxval', eval_maxval, 'DNN_maxval/F')
        DNNCat_branch = output_tree.Branch('DNNCat', DNNCat, 'DNNCat/F')

        sample_name = process
        histoname_type = 'Category'

        histo_ttHclassified_events_title = 'ttH %s Events: %s Sample' % (histoname_type,sample_name)
        histo_ttHclassified_events_name_option1 = 'histo_ttH%s_events_%s_option1' % (histoname_type,sample_name)
        histo_ttHclassified_events_option1 = ROOT.TH1D(histo_ttHclassified_events_name_option1,histo_ttHclassified_events_title,200,0,1.)
        histo_ttJclassified_events_title = 'ttJ %s Events: %s Sample' % (histoname_type,sample_name)
        histo_ttJclassified_events_name_option1 = 'histo_ttJ%s_events_%s_option1' % (histoname_type,sample_name)
        histo_ttJclassified_events_option1 = ROOT.TH1D(histo_ttJclassified_events_name_option1,histo_ttJclassified_events_title,200,0,1.)
        histo_ttWclassified_events_title = 'ttW %s Events: %s Sample' % (histoname_type,sample_name)
        histo_ttWclassified_events_name_option1 = 'histo_ttW%s_events_%s_option1' % (histoname_type,sample_name)
        histo_ttWclassified_events_option1 = ROOT.TH1D(histo_ttWclassified_events_name_option1,histo_ttWclassified_events_title,200,0,1.)
        histo_ttZclassified_events_title = 'ttZ %s Events: %s Sample' % (histoname_type,sample_name)
        histo_ttZclassified_events_name_option1 = 'histo_ttZ%s_events_%s_option1' % (histoname_type,sample_name)
        histo_ttZclassified_events_option1 = ROOT.TH1D(histo_ttZclassified_events_name_option1,histo_ttZclassified_events_title,200,0,1.)

        temp_percentage_done = 0

        #Loop over ttree
        for i in range(data_tree.GetEntries()):
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

            if 'ttH_' in process:
                true_process.append(0)
            if 'Conv' in process or 'Fakes' in process or 'Flips' in process:
                true_process.append(1)
            if 'ttW' in process:
                true_process.append(2)
            if 'ttZ' in process:
                true_process.append(3)

            EventWeights_.append(EventWeight_)

            evaluated_node_values = DNN_applier.evaluate_model(eventnum_resultsprob_dict,Eventnum_)
            event_classification = evaluated_node_values.index(max(evaluated_node_values))
            model1_pred_process.append(event_classification)
            categorised_values = DNN_applier.event_categorised_max_value(event_classification, evaluated_node_values)

            eval_ttHnode_all[0] = evaluated_node_values[0]
            eval_ttJnode_all[0] = evaluated_node_values[1]
            eval_ttWnode_all[0] = evaluated_node_values[2]
            eval_ttZnode_all[0] = evaluated_node_values[3]
            eval_ttHnode_cat[0] = categorised_values[0]
            eval_ttJnode_cat[0] = categorised_values[1]
            eval_ttWnode_cat[0] = categorised_values[2]
            eval_ttZnode_cat[0] = categorised_values[3]
            DNNCat[0] = event_classification
            eval_maxval[0] = evaluated_node_values[event_classification]

            if event_classification == 0:
                histo_ttHclassified_events_option1.Fill(eval_maxval[0],EventWeight_)
            elif event_classification == 1:
                histo_ttJclassified_events_option1.Fill(eval_maxval[0],EventWeight_)
            elif event_classification == 2:
                histo_ttWclassified_events_option1.Fill(eval_maxval[0],EventWeight_)
            elif event_classification == 3:
                histo_ttZclassified_events_option1.Fill(eval_maxval[0],EventWeight_)

            ttH_branch_cat.Fill()
            ttJ_branch_cat.Fill()
            ttW_branch_cat.Fill()
            ttZ_branch_cat.Fill()
            ttH_branch_all.Fill()
            ttJ_branch_all.Fill()
            ttW_branch_all.Fill()
            ttZ_branch_all.Fill()
            DNNmaxval_branch.Fill()
            if eval_maxval < 0.1:
                print 'eval_maxval = ', eval_maxval
            DNNCat_branch.Fill()

        print 'Writing to output file : %s ' % (output_file_name)
        output_file.Write()
        print 'Delete syncTree'
        gDirectory.Delete("syncTree;*")
        print 'Close file'
        output_file.Close()
        print 'Closed'

    plots_dir = os.path.join(samples_final_path_dir,'plots/')

    Plotter.conf_matrix(true_process,model1_pred_process,EventWeights_,'')
    Plotter.save_plots(dir=plots_dir, filename='yields_non_norm_confusion_matrix_APPL.png')
    Plotter.conf_matrix(true_process,model1_pred_process,EventWeights_,'index')
    Plotter.save_plots(dir=plots_dir, filename='yields_norm_confusion_matrix_APPL.png')

    exit(0)

main()
