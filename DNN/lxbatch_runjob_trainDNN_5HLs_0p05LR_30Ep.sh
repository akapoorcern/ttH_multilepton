#!/bin/bash
# Script to run on lxbatch
WORKING_DIR=/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN
cd ${WORKING_DIR}
pwd
scl enable python27 bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-slc6-gcc62-opt/setup.sh
source /afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/lxplus_setup/py2_virtualenv/bin/activate
python train_DNN.py -s samples/2017_updated_MC/TrainMVA/ttH_TrainMVA.root -x samples/2017_updated_MC/TrainMVA/TT_TrainMVA.root -y samples/2017_updated_MC/TrainMVA/ttV.root -a relu -l 5 -t D,G -j input_variables_list_2017samples.json -r 0.05 -n 30
