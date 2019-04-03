
#              train_DNN.py
###         Joshuha Thomas-Wilsker, IHEP Beijing CERN

Python script using Keras with TensorFlow
backend to train deep neural network for
ttH multilepton dilepton analysis region.

## Introduction
- Using pyMVA Keras interface with Tensorflow backend to design DNN model for ttH multilpeton dilepton analysis region.
- In this setup we can easily change the backend in case wewant to use e.g. Theano instead of TensorFlow.
- Import all Keras libraries along with TensorFlow backend into pyMVA so that we can easily produce .root files.
- Using objects from TMVA library to manage the data / book methods etc.
- .root files output with training histograms/ttrees.
- '.h5' file (stores multidimensional arrays of scientific data) stores model.
- pyKeras much faster and better performing (integral of ROC) than TMVA DNN especially when going to higher number of hidden layers.

## New shell
When you first open a new shell you only need to run the commands to source a software stack. This can be e.g. :
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-slc6-gcc62-opt/setup.sh
```
However, the above will only work if you are logged on to an lxplus SLC6 machine. The default lxplus operating system is now CC7 hence look for a CC7 software stack.

## DNN Training
The first script to run is train_DNN.py. This script uses keras to train and test a DNN using Tensorflow. The input arguments are described below and more information can be found in the code:
```
python train-DNN.py -t <train and plot=1, plot only=0> -r <SigRegion/CtrlRegion> -w <class_weights> -l <lepton_selection>
```
e.g.:
```
python train-DNN.py -t 1 -r SigRegion -w InverseSRYields -l loose
```
4 ntuples (ttH(ML) signal, tt+jets background, tt+W background, tt+Z background) containing events from the ttH multilepton analysis training regions should be loaded. Check the current default paths in the code for where the code expects to find the files. The ntuples are converted to dataframes and then .csv files to be read by keras.

Global event weights are set in order to focus the training on a particular sample of events. Any event weights required can also be set in the arguments. The model is built using the keras interface and saved to a .h5 file inside a directory that is named according to the arguments you pass and the hardcoded directory name.

## DNN Application
Inside the `application` directory one can find two scripts used to apply the networks trained in the previous step. The first script evaluates the network and adds the output to the TTrees and also creates the confusion matrices for the network application region:
```
DNN_application.py
```
These scripts will apply the network to events from files in the directory hard coded in the script (can be read from EOS). The code currently evaluates 3 models simultaneously and adds the results to the existing TTrees but this can be changed/updated to suit ones need. The 3 models/networks used are hardcoded but this can also be updated to suit ones needs. Examples on how to run this script can be seen in the run_application_XXXX.sh scripts.

The second script makes further diagnostics plots such as the stacked node response plots:
```
DNN_Evaluation_Control_Plots.py
```
This script acn be run using the following example:
```
python DNN_Evaluation_Control_Plots.py -r SigRegion -d 0 -l loose
```
For more information on the input arguments, one can check the code or use the -h option
