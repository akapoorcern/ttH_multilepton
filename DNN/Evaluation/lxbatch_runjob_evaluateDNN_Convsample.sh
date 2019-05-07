#!/bin/bash
WORKING_DIR=/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/Evaluation
cd ${WORKING_DIR}
source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-slc6-gcc62-opt/setup.sh
python evaluate_DNN.py -j /afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/Evaluation/input_variables_Nov2018.json -s 2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs -i /afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/samples/samples-2018-09-25/2L/Conv_2L.root
