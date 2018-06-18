#!/bin/bash
# Script to run on lxbatch
WORKING_DIR=/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/Evaluation
cd ${WORKING_DIR}
pwd
scl enable python27 bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-slc6-gcc62-opt/setup.sh
source /afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/lxplus_setup/py2_virtualenv/bin/activate
python evaluate_DNN.py -j /afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/input_variables_list_allJets-2018-06-15.json -s 2HLs_relu_D+G-VarTrans_0.008-learnRate_40-epochs -i /afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/samples/2017_updated_MC/2LSS/TTH_2LSS.root
