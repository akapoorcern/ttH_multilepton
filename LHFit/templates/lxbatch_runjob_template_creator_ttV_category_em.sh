#!/bin/bash
WORKING_DIR=/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/LHFit/templates
cd ${WORKING_DIR}
python shape_template_creator.py -j /afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/LHFit/templates/full_samples_list_17Vars.json -o /afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/LHFit/templates/DNN_templates_2018_09_04 -b ttH_2LSS_em_ttVCat
