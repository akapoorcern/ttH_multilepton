# ttH Multilepton LHFit (2LSS)
## Joshuha Thomas-Wilsker


### templates directory
This directory contains all the code needed to create templates for the CMS combine tool, when provided with correctly formatted root files. The script that actually creates the templates is:
```
shape_template_creator.py
```
This runs on root files that have a branch containing the DNN already appended. One can find these files e.g. here:
```
/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/DNN/Evaluation/MultiClass_DNN_21Vars_2HLs_relu_D+G-VarTrans_0.008-learnRate_15-epochs/outputs/2LSS/
```
Note: these scripts are basically the stage in the analysis after the 'DNN/Evaluation' scripts are used.

#### lxbatch Scripts
In templates directory you will find the script:
```
lxbatch_submit_template_creator.sh
```
which when run will submit  several jobs to lxbatch nodes. Each job will produce the template for 1 analysis bin.

Use lxbatch scripts to submit script that creates templates for fit to lxbatch. One thing to be careful of here is that when one submits to the batch node, the local environment is shipped with the job (e.g. anything you've sourced) hence cannot source LCG builds in shell you want to submit from.
