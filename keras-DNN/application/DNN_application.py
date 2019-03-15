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
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.layers import Dropout
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from root_numpy import root2array, tree2array
seed = 7
np.random.seed(7)
rng = np.random.RandomState(31337)

def getEOSlslist(directory, mask='', prepend='root://eosuser.cern.ch/'):
    from subprocess import Popen, PIPE

    eos_cmd = '/afs/cern.ch/project/eos/installation/0.3.15/bin/eos.select'
    eos_dir = '/eos/user/%s ' % (directory)
    data = Popen([eos_cmd, prepend, ' ls ', eos_dir], stdout=PIPE)
    out,err = data.communicate()

    full_list = []

    ## if input file was single root file:
    if directory.endswith('.root'):
        if len(out.split('\n')[0]) > 0:
            return [os.path.join(prepend,eos_dir).replace(" ","")]

    ## instead of only the file name append the string to open the file in ROOT
    for line in out.split('\n'):
        if len(line.split()) == 0: continue
        full_list.append(os.path.join(prepend,eos_dir,line).replace(" ",""))

    ## strip the list of files if required
    if mask != '':
        stripped_list = [x for x in full_list if mask in x]
        return stripped_list

    ## return
    return full_list

def load_data(inputPath,variables,criteria,key,fileName):

    my_cols_list=variables+['corrected_avg_dr_jet']
    data = pd.DataFrame(columns=my_cols_list)

    inputlist = getEOSlslist(directory=inputPath+fileName+".root")
    current_file = str(inputlist[0])
    print 'current_file: ', current_file

    try: tfile = ROOT.TFile(current_file)
    except :
        print " file "+ current_file+" doesn't exist "
        exit(0)
    try: nominal_tree = gDirectory.Get( 'syncTree' )
    except :
        print "Tree doesn't exist in " + current_file
        exit(0)
    if nominal_tree is not None :
        try: chunk_arr = tree2array(tree=nominal_tree, selection=criteria) #,  start=start, stop = stop)
        except :
            print 'Couldnt convert to array'
            exit(0)
        else :
            chunk_df = pd.DataFrame(chunk_arr, columns=variables)
            chunk_df['corrected_avg_dr_jet'] = chunk_df['avg_dr_jet'].multiply((chunk_df['Jet_numLoose'] - 1), axis="index")
            chunk_df['corrected_avg_dr_jet'] = chunk_df[['corrected_avg_dr_jet']].divide(2)
            data=data.append(chunk_df, ignore_index=True)
    else:
        print 'nominal_tree none existent'
    tfile.Close()
    '''if len(data) == 0 :
        print 'No data! Exiting.'
        exit(0)'''
    return data

def baseline_model(num_variables,optimizer):
    model = Sequential()
    # input_dim = number of variables
    model.add(Dense(32,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    for index in xrange(5):
        model.add(Dense(16,activation='relu'))
    for index in xrange(5):
        model.add(Dense(16,activation='relu'))
    for index in xrange(5):
        model.add(Dense(8,activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=["acc"])
    return model


def load_trained_model(weights_path, num_variables, optimizer):
    print 'loading weights_path: ', weights_path
    model = baseline_model(num_variables, optimizer)
    model.load_weights(weights_path)
    from keras.models import load_model
    return model

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def main():

    number_of_classes = 4

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)

    parser.add_argument('-p', '--processName', dest='processName', help='Process name. List of options in keys of process_filename dictionary', default=[], type=str, nargs='+')
    parser.add_argument('-r', '--region', dest='region', help='Option to choose SigRegion or CtrlRegion', default='SigRegion', type=str)
    parser.add_argument('-j', '--JES', dest='JES', help='Option to choose whether to run on JES Syst samples (0=Nominal, 1=JESUp, 2=JESDown)', default=0, type=int)
    parser.add_argument('-l', '--lepsel', dest='lepsel', help='type of lepton selection for trained model (loose, fakeable)', default='loose', type=str)
    args = parser.parse_args()
    processes = args.processName
    region = args.region
    JES_flag = args.JES
    leptonsel = args.lepsel

    print 'region = ', region
    print 'leptonsel = ', leptonsel
    print 'JES_flag = ', JES_flag

    input_name = ''
    input_var_jsonFile = ''
    if region == 'CtrlRegion':
        input_name = '2019-03-13_2OptionMerge_ttWctrl_'+leptonsel
        if JES_flag==1:
            input_name = input_name+'_JESUp'
        if JES_flag==2:
            input_name = input_name+'_JESDown'
        region = 'ttWctrl'
        input_var_jsonFile = open('../input_vars_CtrlRegion.json','r')
    elif region == 'SigRegion':
        input_name = '2019-03-13_2OptionMerge_'+leptonsel
        if JES_flag==1:
            input_name = input_name+'_JESUp'
        if JES_flag==2:
            input_name = input_name+'_JESDown'
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
        if 'avg_dr_jet' in key:
            key = 'corrected_avg_dr_jet'
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

    #model_name = os.path.join('../',input_name,'model.h5')
    #model = load_trained_model(model_name, num_variables, optimizer)
    print 'region = ', region
    input_models_path = ''
    if region == 'SigRegion':
        if leptonsel == 'loose':
            #input_models_path = ['2019-03-07_loose_InverseNEventsTR_SigRegion','2019-03-07_loose_InverseSRYields_SigRegion','2019-03-07_loose_SRYieldsOverNEventsTR_SigRegion','2019-03-07_loose_InverseSumWeightsTR_SigRegion']
            input_models_path = ['2019-03-07_loose_InverseNEventsTR_SigRegion','2019-03-07_loose_InverseSRYields_SigRegion']
        elif leptonsel == 'fakeable':
            #input_models_path = ['2019-03-07_fakeable_InverseNEventsTR_SigRegion','2019-03-07_fakeable_InverseSRYields_SigRegion','2019-03-07_fakeable_SRYieldsOverNEventsTR_SigRegion','2019-03-07_fakeable_InverseSumWeightsTR_SigRegion']
            input_models_path = ['2019-03-07_fakeable_InverseNEventsTR_SigRegion','2019-03-07_fakeable_InverseSRYields_SigRegion']
        elif leptonsel == 'mixed':
                #input_models_path = ['2019-03-13_mixed_InverseNEventsTR_SigRegion','2019-03-13_mixed_InverseSRYields_SigRegion','2019-03-07_fakeable_SRYieldsOverNEventsTR_SigRegion','2019-03-07_fakeable_InverseSumWeightsTR_SigRegion']
                input_models_path = ['2019-03-13_mixed_InverseNEventsTR_SigRegion','2019-03-13_mixed_InverseSRYields_SigRegion']
    elif region == 'ttWctrl':
        if leptonsel == 'loose':
            #input_models_path = ['2019-03-07_loose_InverseNEventsTR_CtrlRegion', '2019-03-07_loose_InverseSRYields_CtrlRegion','2019-03-07_loose_SRYieldsOverNEventsTR_CtrlRegion','2019-03-07_loose_InverseSumWeightsTR_CtrlRegion']
            input_models_path = ['2019-03-07_loose_InverseNEventsTR_CtrlRegion', '2019-03-07_loose_InverseSRYields_CtrlRegion']
        elif leptonsel == 'fakeable':
            #input_models_path = ['2019-03-07_fakeable_InverseNEventsTR_CtrlRegion', '2019-03-07_fakeable_InverseSRYields_CtrlRegion','2019-03-07_fakeable_SRYieldsOverNEventsTR_CtrlRegion','2019-03-07_fakeable_InverseSumWeightsTR_CtrlRegion']
            input_models_path = ['2019-03-07_fakeable_InverseNEventsTR_CtrlRegion', '2019-03-07_fakeable_InverseSRYields_CtrlRegion']

    print input_models_path[0]

    model_name_1 = os.path.join('../',input_models_path[0],'model.h5')
    model_1 = load_trained_model(model_name_1, num_variables, optimizer)
    model_name_2 = os.path.join('../',input_models_path[1],'model.h5')
    model_2 = load_trained_model(model_name_2, num_variables, optimizer)
    '''model_name_3 = os.path.join('../',input_models_path[2],'model.h5')
    model_3 = load_trained_model(model_name_3, num_variables, optimizer)
    model_name_4 = os.path.join('../',input_models_path[3],'model.h5')
    model_4 = load_trained_model(model_name_4, num_variables, optimizer)'''

    Plotter = plotter()

    true_process = []
    option1_pred_process = []
    option2_pred_process = []
    option3_pred_process = []
    option4_pred_process = []

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

        samples_final_path_dir = os.path.join(samples_dir_w_appended_DNN,input_name)
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
            if region == 'SigRegion' or 'JES' in region:
                data = load_data(inputs_file_path,column_headers,'Jet_numLoose>=4',process,process_filename.get(process))
            elif region == 'ttWctrl':
                data = load_data(inputs_file_path,column_headers,'Jet_numLoose==3',process,process_filename.get(process))
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

        result_probs_test = model_1.predict_proba(np.array(X_test))
        result_classes_test = model_1.predict_classes(np.array(X_test))

        result_probs_test_2 = model_2.predict_proba(np.array(X_test))
        result_classes_test_2 = model_2.predict_classes(np.array(X_test))

        '''result_probs_test_3 = model_3.predict_proba(np.array(X_test))
        result_classes_test_3 = model_3.predict_classes(np.array(X_test))

        result_probs_test_4 = model_4.predict_proba(np.array(X_test))
        result_classes_test_4 = model_4.predict_classes(np.array(X_test))'''

        # create dictionary where the value is the array of probabilities for the four categories
        # and the key is the event number.
        eventnum_resultsprob_dict = {}
        eventnum_resultsprob_1_dict = {}
        '''eventnum_resultsprob_2_dict = {}
        eventnum_resultsprob_3_dict = {}'''
        for index in range(result_probs_test.shape[0]):
            eventnum_resultsprob_dict[nEvent[index]] = result_probs_test[index]
            eventnum_resultsprob_1_dict[nEvent[index]] = result_probs_test_2[index]
            '''eventnum_resultsprob_2_dict[nEvent[index]] = result_probs_test_3[index]
            eventnum_resultsprob_3_dict[nEvent[index]] = result_probs_test_4[index]'''

        inputlist = getEOSlslist(directory=inputs_file_path+current_sample_name+".root")
        current_file = str(inputlist[0])

        # Open files and load ttrees
        data_file = TFile.Open(current_file)
        data_tree = data_file.Get("syncTree")

        output_file_name = '%s/%s.root' % (samples_final_path_dir,process_filename.get(process))
        output_file = TFile.Open(output_file_name,'RECREATE')
        output_tree = data_tree.CopyTree("")
        output_tree.SetName("output_tree")
        nEvents_check = output_tree.BuildIndex("nEvent","run")

        eval_ttHnode = array('f',[0.])
        eval_ttJnode = array('f',[0.])
        eval_ttWnode = array('f',[0.])
        eval_ttZnode = array('f',[0.])
        eval_ttHnode_option2 = array('f',[0.])
        eval_ttJnode_option2 = array('f',[0.])
        eval_ttWnode_option2 = array('f',[0.])
        eval_ttZnode_option2 = array('f',[0.])
        eval_maxval = array('f',[0.])
        eval_maxval_2 = array('f',[0.])
        '''eval_maxval_3 = array('f',[0.])
        eval_maxval_4 = array('f',[0.])'''
        DNNCat = array('f',[0.])
        DNNCat_option2 = array('f',[0.])
        '''DNNCat_option3 = array('f',[0.])
        DNNCat_option4 = array('f',[0.])'''
        ttH_branch = output_tree.Branch('DNN_ttHnode', eval_ttHnode, 'DNN_ttHnode/F')
        ttJ_branch = output_tree.Branch('DNN_ttJnode', eval_ttJnode, 'DNN_ttJnode/F')
        ttW_branch = output_tree.Branch('DNN_ttWnode', eval_ttWnode, 'DNN_ttWnode/F')
        ttZ_branch = output_tree.Branch('DNN_ttZnode', eval_ttZnode, 'DNN_ttZnode/F')

        ttH_branch_option2 = output_tree.Branch('DNN_ttHnode_option2', eval_ttHnode_option2, 'DNN_ttHnode_option2/F')
        ttJ_branch_option2 = output_tree.Branch('DNN_ttJnode_option2', eval_ttJnode_option2, 'DNN_ttJnode_option2/F')
        ttW_branch_option2 = output_tree.Branch('DNN_ttWnode_option2', eval_ttWnode_option2, 'DNN_ttWnode_option2/F')
        ttZ_branch_option2 = output_tree.Branch('DNN_ttZnode_option2', eval_ttZnode_option2, 'DNN_ttZnode_option2/F')

        DNNMaxval_branch = output_tree.Branch('DNN_maxval', eval_maxval, 'DNN_maxval/F')
        DNNMaxval_branch_2 = output_tree.Branch('DNN_maxval_option2', eval_maxval_2, 'DNN_maxval_option2/F')
        '''DNNMaxval_branch_3 = output_tree.Branch('DNN_maxval_option3', eval_maxval_3, 'DNN_maxval_option3/F')
        DNNMaxval_branch_4 = output_tree.Branch('DNN_maxval_option4', eval_maxval_4, 'DNN_maxval_option4/F')'''
        DNNCat_branch = output_tree.Branch('DNNCat', DNNCat, 'DNNCat/F')
        DNNCat_branch_option2 = output_tree.Branch('DNNCat_option2', DNNCat_option2, 'DNNCat_option2/F')
        '''DNNCat_branch_option3 = output_tree.Branch('DNNCat_option3', DNNCat_option3, 'DNNCat_option3/F')
        DNNCat_branch_option4 = output_tree.Branch('DNNCat_option4', DNNCat_option4, 'DNNCat_option4/F')'''

        sample_name = process
        histoname_type = 'Category'

        histo_ttHclassified_events_title = 'ttH %s Events: %s Sample' % (histoname_type,sample_name)
        histo_ttHclassified_events_name_option1 = 'histo_ttH%s_events_%s_option1' % (histoname_type,sample_name)
        histo_ttHclassified_events_option1 = ROOT.TH1D(histo_ttHclassified_events_name_option1,histo_ttHclassified_events_title,200,0,1.)
        histo_ttHclassified_events_name_option2 = 'histo_ttH%s_events_%s_option2' % (histoname_type,sample_name)
        histo_ttHclassified_events_option2 = ROOT.TH1D(histo_ttHclassified_events_name_option2,histo_ttHclassified_events_title,200,0,1.)
        '''histo_ttHclassified_events_name_option3 = 'histo_ttH%s_events_%s_option3' % (histoname_type,sample_name)
        histo_ttHclassified_events_option3 = ROOT.TH1D(histo_ttHclassified_events_name_option3,histo_ttHclassified_events_title,200,0,1.)
        histo_ttHclassified_events_name_option4 = 'histo_ttH%s_events_%s_option4' % (histoname_type,sample_name)
        histo_ttHclassified_events_option4 = ROOT.TH1D(histo_ttHclassified_events_name_option4,histo_ttHclassified_events_title,200,0,1.)'''

        histo_ttJclassified_events_title = 'ttJ %s Events: %s Sample' % (histoname_type,sample_name)
        histo_ttJclassified_events_name_option1 = 'histo_ttJ%s_events_%s_option1' % (histoname_type,sample_name)
        histo_ttJclassified_events_option1 = ROOT.TH1D(histo_ttJclassified_events_name_option1,histo_ttJclassified_events_title,200,0,1.)
        histo_ttJclassified_events_name_option2 = 'histo_ttJ%s_events_%s_option2' % (histoname_type,sample_name)
        histo_ttJclassified_events_option2 = ROOT.TH1D(histo_ttJclassified_events_name_option2,histo_ttJclassified_events_title,200,0,1.)
        '''histo_ttJclassified_events_name_option3 = 'histo_ttJ%s_events_%s_option3' % (histoname_type,sample_name)
        histo_ttJclassified_events_option3 = ROOT.TH1D(histo_ttJclassified_events_name_option3,histo_ttJclassified_events_title,200,0,1.)
        histo_ttJclassified_events_name_option4 = 'histo_ttJ%s_events_%s_option4' % (histoname_type,sample_name)
        histo_ttJclassified_events_option4 = ROOT.TH1D(histo_ttJclassified_events_name_option4,histo_ttJclassified_events_title,200,0,1.)'''

        histo_ttWclassified_events_title = 'ttW %s Events: %s Sample' % (histoname_type,sample_name)
        histo_ttWclassified_events_name_option1 = 'histo_ttW%s_events_%s_option1' % (histoname_type,sample_name)
        histo_ttWclassified_events_option1 = ROOT.TH1D(histo_ttWclassified_events_name_option1,histo_ttWclassified_events_title,200,0,1.)
        histo_ttWclassified_events_name_option2 = 'histo_ttW%s_events_%s_option2' % (histoname_type,sample_name)
        histo_ttWclassified_events_option2 = ROOT.TH1D(histo_ttWclassified_events_name_option2,histo_ttWclassified_events_title,200,0,1.)
        '''histo_ttWclassified_events_name_option3 = 'histo_ttW%s_events_%s_option3' % (histoname_type,sample_name)
        histo_ttWclassified_events_option3 = ROOT.TH1D(histo_ttWclassified_events_name_option3,histo_ttWclassified_events_title,200,0,1.)
        histo_ttWclassified_events_name_option4 = 'histo_ttW%s_events_%s_option4' % (histoname_type,sample_name)
        histo_ttWclassified_events_option4 = ROOT.TH1D(histo_ttWclassified_events_name_option4,histo_ttWclassified_events_title,200,0,1.)
        '''
        histo_ttZclassified_events_title = 'ttZ %s Events: %s Sample' % (histoname_type,sample_name)
        histo_ttZclassified_events_name_option1 = 'histo_ttZ%s_events_%s_option1' % (histoname_type,sample_name)
        histo_ttZclassified_events_option1 = ROOT.TH1D(histo_ttZclassified_events_name_option1,histo_ttZclassified_events_title,200,0,1.)
        histo_ttZclassified_events_name_option2 = 'histo_ttZ%s_events_%s_option2' % (histoname_type,sample_name)
        histo_ttZclassified_events_option2 = ROOT.TH1D(histo_ttZclassified_events_name_option2,histo_ttZclassified_events_title,200,0,1.)
        '''histo_ttZclassified_events_name_option3 = 'histo_ttZ%s_events_%s_option3' % (histoname_type,sample_name)
        histo_ttZclassified_events_option3 = ROOT.TH1D(histo_ttZclassified_events_name_option3,histo_ttZclassified_events_title,200,0,1.)
        histo_ttZclassified_events_name_option4 = 'histo_ttZ%s_events_%s_option4' % (histoname_type,sample_name)
        histo_ttZclassified_events_option4 = ROOT.TH1D(histo_ttZclassified_events_name_option4,histo_ttZclassified_events_title,200,0,1.)
        '''
        temp_percentage_done = 0
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

            event_classification = max(eventnum_resultsprob_dict.get(Eventnum_)[0], eventnum_resultsprob_dict.get(Eventnum_)[1], eventnum_resultsprob_dict.get(Eventnum_)[2], eventnum_resultsprob_dict.get(Eventnum_)[3])
            event_classification_2 = max(eventnum_resultsprob_1_dict.get(Eventnum_)[0], eventnum_resultsprob_1_dict.get(Eventnum_)[1], eventnum_resultsprob_1_dict.get(Eventnum_)[2], eventnum_resultsprob_1_dict.get(Eventnum_)[3])
            '''event_classification_3 = max(eventnum_resultsprob_2_dict.get(Eventnum_)[0], eventnum_resultsprob_2_dict.get(Eventnum_)[1], eventnum_resultsprob_2_dict.get(Eventnum_)[2], eventnum_resultsprob_2_dict.get(Eventnum_)[3])
            event_classification_4 = max(eventnum_resultsprob_3_dict.get(Eventnum_)[0], eventnum_resultsprob_3_dict.get(Eventnum_)[1], eventnum_resultsprob_3_dict.get(Eventnum_)[2], eventnum_resultsprob_3_dict.get(Eventnum_)[3])'''

            if 'ttH_' in process:
                true_process.append(0)
            if 'Conv' in process or 'Fakes' in process or 'Flips' in process:
                true_process.append(1)
            if 'ttW' in process:
                true_process.append(2)
            if 'ttZ' in process:
                true_process.append(3)

            if event_classification == eventnum_resultsprob_dict.get(Eventnum_)[0]:
                option1_pred_process.append(0)
                eval_ttHnode[0] = eventnum_resultsprob_dict.get(Eventnum_)[0]
                eval_ttJnode[0] = -2
                eval_ttWnode[0] = -2
                eval_ttZnode[0] = -2
                histo_ttHclassified_events_option1.Fill(event_classification,EventWeight_)
                DNNCat[0] = 1
                eval_maxval[0] = eventnum_resultsprob_dict.get(Eventnum_)[0]
            elif event_classification == eventnum_resultsprob_dict.get(Eventnum_)[1]:
                option1_pred_process.append(1)
                eval_ttJnode[0] = eventnum_resultsprob_dict.get(Eventnum_)[1]
                eval_ttHnode[0] = -2
                eval_ttWnode[0] = -2
                eval_ttZnode[0] = -2
                histo_ttJclassified_events_option1.Fill(event_classification,EventWeight_)
                DNNCat[0] = 2
                eval_maxval[0] = eventnum_resultsprob_dict.get(Eventnum_)[1]
            elif event_classification == eventnum_resultsprob_dict.get(Eventnum_)[2]:
                option1_pred_process.append(2)
                eval_ttWnode[0] = eventnum_resultsprob_dict.get(Eventnum_)[2]
                eval_ttHnode[0] = -2
                eval_ttJnode[0] = -2
                eval_ttZnode[0] = -2
                histo_ttWclassified_events_option1.Fill(event_classification,EventWeight_)
                DNNCat[0] = 3
                eval_maxval[0] = eventnum_resultsprob_dict.get(Eventnum_)[2]
            elif event_classification == eventnum_resultsprob_dict.get(Eventnum_)[3]:
                option1_pred_process.append(3)
                eval_ttZnode[0] = eventnum_resultsprob_dict.get(Eventnum_)[3]
                eval_ttHnode[0] = -2
                eval_ttJnode[0] = -2
                eval_ttWnode[0] = -2
                histo_ttZclassified_events_option1.Fill(event_classification,EventWeight_)
                DNNCat[0] = 4
                eval_maxval[0] = eventnum_resultsprob_dict.get(Eventnum_)[3]

            if event_classification_2 == eventnum_resultsprob_1_dict.get(Eventnum_)[0]:
                option2_pred_process.append(0)
                eval_ttHnode_option2[0] = eventnum_resultsprob_1_dict.get(Eventnum_)[0]
                eval_ttJnode_option2[0] = -2
                eval_ttWnode_option2[0] = -2
                eval_ttZnode_option2[0] = -2
                DNNCat_option2[0] = 1
                eval_maxval_2[0] = eventnum_resultsprob_1_dict.get(Eventnum_)[0]
                histo_ttHclassified_events_option2.Fill(event_classification_2,EventWeight_)
            elif event_classification_2 == eventnum_resultsprob_1_dict.get(Eventnum_)[1]:
                option2_pred_process.append(1)
                eval_ttJnode_option2[0] = eventnum_resultsprob_1_dict.get(Eventnum_)[1]
                eval_ttHnode_option2[0] = -2
                eval_ttWnode_option2[0] = -2
                eval_ttZnode_option2[0] = -2
                DNNCat_option2[0] = 2
                eval_maxval_2[0] = eventnum_resultsprob_1_dict.get(Eventnum_)[1]
                histo_ttJclassified_events_option2.Fill(event_classification_2,EventWeight_)
            elif event_classification_2 == eventnum_resultsprob_1_dict.get(Eventnum_)[2]:
                option2_pred_process.append(2)
                eval_ttWnode_option2[0] = eventnum_resultsprob_1_dict.get(Eventnum_)[2]
                eval_ttHnode_option2[0] = -2
                eval_ttJnode_option2[0] = -2
                eval_ttZnode_option2[0] = -2
                DNNCat_option2[0] = 3
                eval_maxval_2[0] = eventnum_resultsprob_1_dict.get(Eventnum_)[2]
                histo_ttWclassified_events_option2.Fill(event_classification_2,EventWeight_)
            elif event_classification_2 == eventnum_resultsprob_1_dict.get(Eventnum_)[3]:
                option2_pred_process.append(3)
                eval_ttZnode_option2[0] = eventnum_resultsprob_1_dict.get(Eventnum_)[3]
                eval_ttHnode_option2[0] = -2
                eval_ttJnode_option2[0] = -2
                eval_ttWnode_option2[0] = -2
                DNNCat_option2[0] = 4
                eval_maxval_2[0] = eventnum_resultsprob_1_dict.get(Eventnum_)[3]
                histo_ttZclassified_events_option2.Fill(event_classification_2,EventWeight_)

            '''if event_classification_3 == eventnum_resultsprob_2_dict.get(Eventnum_)[0]:
                option3_pred_process.append(0)
                DNNCat_option3[0] = 1
                eval_maxval_3[0] = eventnum_resultsprob_2_dict.get(Eventnum_)[0]
                histo_ttHclassified_events_option3.Fill(event_classification_3,EventWeight_)
            elif event_classification_3 == eventnum_resultsprob_2_dict.get(Eventnum_)[1]:
                option3_pred_process.append(1)
                DNNCat_option3[0] = 2
                eval_maxval_3[0] = eventnum_resultsprob_2_dict.get(Eventnum_)[1]
                histo_ttJclassified_events_option3.Fill(event_classification_3,EventWeight_)
            elif event_classification_3 == eventnum_resultsprob_2_dict.get(Eventnum_)[2]:
                option3_pred_process.append(2)
                DNNCat_option3[0] = 3
                eval_maxval_3[0] = eventnum_resultsprob_2_dict.get(Eventnum_)[2]
                histo_ttWclassified_events_option3.Fill(event_classification_3,EventWeight_)
            elif event_classification_3 == eventnum_resultsprob_2_dict.get(Eventnum_)[3]:
                option3_pred_process.append(3)
                DNNCat_option3[0] = 4
                eval_maxval_3[0] = eventnum_resultsprob_2_dict.get(Eventnum_)[3]
                histo_ttZclassified_events_option3.Fill(event_classification_3,EventWeight_)


            if event_classification_4 == eventnum_resultsprob_3_dict.get(Eventnum_)[0]:
                option4_pred_process.append(0)
                DNNCat_option4[0] = 1
                eval_maxval_4[0] = eventnum_resultsprob_3_dict.get(Eventnum_)[0]
                histo_ttHclassified_events_option4.Fill(event_classification_4,EventWeight_)
            elif event_classification_4 == eventnum_resultsprob_3_dict.get(Eventnum_)[1]:
                option4_pred_process.append(1)
                DNNCat_option4[0] = 2
                eval_maxval_4[0] = eventnum_resultsprob_3_dict.get(Eventnum_)[1]
                histo_ttJclassified_events_option4.Fill(event_classification_4,EventWeight_)
            elif event_classification_4 == eventnum_resultsprob_3_dict.get(Eventnum_)[2]:
                option4_pred_process.append(2)
                DNNCat_option4[0] = 3
                eval_maxval_4[0] = eventnum_resultsprob_3_dict.get(Eventnum_)[2]
                histo_ttWclassified_events_option4.Fill(event_classification_4,EventWeight_)
            elif event_classification_4 == eventnum_resultsprob_3_dict.get(Eventnum_)[3]:
                option4_pred_process.append(3)
                DNNCat_option4[0] = 4
                eval_maxval_4[0] = eventnum_resultsprob_3_dict.get(Eventnum_)[3]
                histo_ttZclassified_events_option4.Fill(event_classification_4,EventWeight_)'''

            ttH_branch.Fill()
            ttJ_branch.Fill()
            ttW_branch.Fill()
            ttZ_branch.Fill()
            ttH_branch_option2.Fill()
            ttJ_branch_option2.Fill()
            ttW_branch_option2.Fill()
            ttZ_branch_option2.Fill()
            DNNMaxval_branch.Fill()
            DNNMaxval_branch_2.Fill()
            #DNNMaxval_branch_3.Fill()
            #DNNMaxval_branch_4.Fill()
            DNNCat_branch.Fill()
            DNNCat_branch_option2.Fill()
            #DNNCat_branch_option3.Fill()
            #DNNCat_branch_option4.Fill()

        print 'Writing to output file. . . . '
        output_file.Write()
        print 'Delete syncTree'
        gDirectory.Delete("syncTree;*")
        print 'Close file'
        output_file.Close()
        print 'Closed'

    plots_dir = os.path.join(samples_final_path_dir,'plots/')
    Plotter.conf_matrix(true_process,option1_pred_process,'index')
    Plotter.save_plots(dir=plots_dir, filename='confusion_acc_matrix_option1.png')
    Plotter.conf_matrix(true_process,option1_pred_process,'columns')
    Plotter.save_plots(dir=plots_dir, filename='confusion_purity_matrix_option1.png')
    Plotter.conf_matrix(true_process,option2_pred_process,'index')
    Plotter.save_plots(dir=plots_dir, filename='confusion_acc_matrix_option2.png')
    Plotter.conf_matrix(true_process,option2_pred_process,'columns')
    Plotter.save_plots(dir=plots_dir, filename='confusion_purity_matrix_option2.png')
    '''Plotter.conf_matrix(true_process,option3_pred_process)
    Plotter.save_plots(dir=plots_dir, filename='confusion_matrix_option3.png')
    Plotter.conf_matrix(true_process,option4_pred_process)
    Plotter.save_plots(dir=plots_dir, filename='confusion_matrix_option4.png')'''
    exit(0)
main()
