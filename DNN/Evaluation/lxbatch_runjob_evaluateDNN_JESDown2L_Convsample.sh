#!/bin/bash
WORKING_DIR=/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/Evaluation
cd ${WORKING_DIR}
source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-slc6-gcc62-opt/setup.sh
python evaluate_DNN.py -j /afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/input_variables_list-2018-11-13.json -s 35Vars_2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs-40-nodes -i /afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/samples/samples-2018-11-09/JESDown2L/Conv_JESDown2L.root
