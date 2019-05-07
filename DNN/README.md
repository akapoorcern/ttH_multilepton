
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


## Initial setup
The fermilab keras workshop has a nice set of scripts one can use to setup the keras environment. Checkout the package:
```
git clone https://github.com/stwunsch/fermilab_keras_workshop
```
and change into the cloned repository. On lxplus we needed python 2.7 for initial setup. Check which python versions are available:
```
scl -l | grep python
```
Enable the one you want:
```
scl enable python27 bash
```
Run the following command to install necessary packages with pip:
```
bash init_virtualenv.sh
```
The bash script will set up the py2_virtualenv directory. It then sources the py2_virtualenv/bin/activate which simply activates the python virtual environment. This last step will put the py2_virtualenv directory into your current directory and set up the relevant packages in your virtual environment.

Because this script uses pyRoot/pyMVA enabled we need an additional step to so that the root libraries are accessible (apologies for sourcing a nightly, currently only version with tensor_forest etc.):
```
(stable LCG build) source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-slc6-gcc62-opt/setup.sh
(latest LCG build) source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev3/latest/x86_64-slc6-gcc62-opt/setup.sh
(nightly LCG build) source /cvmfs/sft-nightlies.cern.ch/lcg/views/dev3/latest/x86_64-slc6-gcc62-opt/setup.sh
```
The following instructions should only be applied if you notice errors with the 'werkzeug' or 'tensor_forest' packagea. These are temporary as they will be included in next LCG build. For now, we have to install 'werkzeug' and correctly link 'tensor_forest' plugins:
```
pip install --user werkzeug
```
and
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/sft-nightlies.cern.ch/lcg/views/dev3/latest/x86_64-slc6-gcc62-opt/lib/python2.7/site-packages/tensorflow/contrib/tensor_forest/
```

## New shell
As mentioned above, now every time you open a new shell you only need to rerun the commands beneath to establish you python working environment:
Source the following software stack:
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-slc6-gcc62-opt/setup.sh
```

## DNN Training
The first script to run is train_DNN.py. This script uses the keras interface within pyTMVA to train and test a DNN using Tensorflow. The input arguments are described below and more information can be found in the code:
```
python train_DNN.py -s <relative_path_to_signal_sample/sample>.root -x <relative_path_to_bckg1_sample/sample>.root -y <relative_path_to_bckg2_sample/sample>.root -a <activation_function> -l <number_of_hidden_layers> -t <input_variable_transformation> -j <variables_list>.json -r <learning_rate> -n <number_of_epochs>
```

Three ntuples (ttH(ML) signal, tt+V background, tt+jets background) containing events from the ttH multilepton analysis training regions should be loaded. The files you wish to load can be passed as command line inputs. Check the current default paths in the code for where the code expects to find the files.

One can also pass as arguments the activation function, number of hidden layers and a .json list of variables. This should make it easier to perform network optimisation studies. The TMVA factory object uses the arguments passed to the script to create the directory where the weights are stored that should inform the user of the variable hyperparameters used for the network architecture.

To perform network optimisation studies, use the DNN-training-helmsman.sh. Here you will find multiple command lines that will train various networks. Each command line should represent a different training e.g. different input list or different number of hidden layers. Another set of scripts will use the various networks to evaluate how the performance changes when varying a specific hyperparameter.

Global event weights are set in order to focus the training on a particular sample of events. Any event weights required are also set in the training script. The model is built using the keras interface and saved to a .h5 file.

The factory object books and uses the model for training, testing and evaluation of the methods. The outputs from this are contained in a .root file in the working dircetory whereas the weights are stored in a .xml which will be inside a directory with the same name as the dataloader object and the file will have the name of the factory.

## Normalisation
- One very important parameter here is 'NormMode'.
  - 'NormMode=EqualNumEvents' : Sets the average weight of signal events to 1. The number of effective events in the signal(/first) class are then normalised to equal the sum of all the other(/background) event weights.
  - 'NormMode=NumEvents' : Average weight of 1 per event, signal and background re-normalised independently. All classes have the same number of effective events.
- In the case of ttH multilepton, if one *doesn't* normalise background to the same as signal, the DNN will separate out ttV seeing as the MC has many more events (DNN tunes to this).

## Number of Hidden layers
- To study how the number of hidden layers affects the output distributions use the helmsman to run multiple trainings changing the number of hidden layers each time.
- On last inspection, increasing the number of hidden layers seems to make ttJets more background like, ttW more signal like and ttH remains in the middle (if perhaps more background like).

## DNN Outputs
- The analysis performed is a 'multiclass' analysis which has three output nodes.
- 'softmax' activation function used in final layer so that the outputs represent probabilities (output is normalised to 1).
- A consequence of the softmax function is that the probability of a class is not independent of the other class probabilities.
- Fine if we want to select a single label per sampling.
- In this case, for a given event, you get three probabilities, one for each output node.
- We assign the event to the label associated with the output node with the highest probability.
- Number of neurons in the output layer should be the same as the number of samples[processes] you want to run.
- When running in TMVA using Keras interface you should see something like:

```
PrepareTrainingAndTestTree
                       : Dataset[MultiClass_DNN] : Class index : 0  name : sample=ttH
                       : Dataset[MultiClass_DNN] : Class index : 1  name : sample=ttV
                       : Dataset[MultiClass_DNN] : Class index : 2  name : sample=ttJets
```
- As you can see here, the three classes (neurons) in the output layer are associated with one of the inputs.


## Checking Models
- If you ever need to check the network model of a training you can check in the /weights/<Factory object>.C


## DNN Training/Testing Plots
- Various plots from the training/testing of the DNN can be created using the appropriately titled script DNN_ResponsePlotter.py.
- The script takes the .root file from the training script as input and makes plots of the combined response from all the output layer nodes along with plots of the individual nodes.
- Output plots are stored in the 'plots' sub-directory of the training directory for example 'MultiClass_DNN_<parameters_of_network>/plots'.
- The individual histograms are store in an output .root file whereas the canvas' of the plots are drawn into .pdf files normally titled 'MCDNN_Response_<network_parameters>.pdf'.
- For example, if you created a file title 'MultiClass_DNN_2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs.root' via the training script one can obtain the response and overtraining distributions by running the command:
```
python DNN_ResponsePlotter.py -s <Suffix_of_training_directory>
```
So for example, one might use the following command:
```
python DNN_ResponsePlotter.py -s 2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs
```
- The script will finds the .root one created during training, creates a 'plots' directory in the same directory as the TMVA factory put the weights files (e.g. 'MultiClass_DNN_2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs/' ) and places the distributions there.

## Kolomogrov-Smirnov test
- The script also produces an overtraining plot. This is a plot of the DNN output distribution on each of the nodes. A distribution is obtain from the training and testing sample and overlaid. A ratio plot of the two distributions is added to the bottom of the canvas and the result from the two-sample Kolomogrov-Smirnov test is written on the main plot.
- Using Kolomogrov-Smirnov as data distributions are non-gaussian we use the KS-test.
- KS test is most sensitive when empirical distribution functions differ in a global fashion near the center of the distribution.
- However if there are repeated deviations between the EDF's or they have the same mean values then the EDFs cross each other multiple times and the maximum deviation between the distributions is reduced.
- The two-sample Kolmogrov-Smirnov test from the scipy package returns two values. The first value is the K-S test statistic and the second value is the 'p-value'. The K-S test statistic gives the supremum deviation between the two distributions and the p-value.
- The p(probability)-value is the probability of getting a more extreme observation (more extreme statistic) than the one observed based on the assumed statistical hypothesis (both distributions sampled from the same probability distribution) is true. For low p-values we conclude the null hypothesis is rejected. One can choose how to define 'low' for example typically a p-value <=0.05 is required to reject the null hypothesis.
- p-values are often used to calculate the significance. The mathematical definition of significance is '-log(p-value)' hence for small p-values (reject the null hypothesis) we get larger significances (as p -> 0, sig -> -inf).

WARNING:
If you see the following warning:
```
approximate p-value will be computed by extrapolation
```
take test result and p-value with pinch of salt.

## Anderson-Darling test
- Stronger test when differences in distributions are near beginning or end of distributions.
- Return values:
  - Statistic: Normalized k-sample Anderson-Darling test statistic.
  - Critical values: the critical values for significance levels: 25%, 10%, 5%, 2.5%, 1%
  - Significance level: An approximate significance level at which the null hypothesis for the provided samples can be rejected.
- For example, if we see the following:
```
(statistic=4.4797806271353506, critical_values=array([ 0.49854918,  1.3236709 ,  1.91577682,  2.49304213,  3.24593219]), significance_level=0.0020491057074350956)
```
- The significance is 0.002 we conclude that the samples are drawn from diffrent populations as the significance is smaller than the 25% critical value.

- Currently this code *does not* use the Anderson-Darling test. This is due to an issue with the scipy implementation of the test.
- For example we see the following result comparing the ttH node test/train distributions:
```
Anderson_ksampResult(statistic=-1.2759636700023367, critical_values=array([ 0.325,  1.226,  1.961,  2.718,  3.752]), significance_level=1.3679528987986334)
```
- The significance level is above 1. This is due to inaccurate extrapolation of the significance to regions outside the critical values.

## Input Variable plots and Separation
- Plots of the input variables and a file containing values of the separation of each of the input processes for said variable are made available using the DNN_InputVariable_Separation.py script.
- The plots of each of the variables will be placed inside the 'plots' sub-directory of the DNN training directory.
- The file containing the separation will be placed inside the 'data' sub-directory of the DNN training directory.
- One can run the script using the following example command:
```
python DNN_InputVariable_Separation.py -s <Suffix_of_training_directory> -j <input_variables_list>.json
```
So following on from the previous examples one might do:
```
DNN_InputVariable_Separation.py -s 2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs -j input_variables_list.json
```

## ROC Curves


- The DNN_ROCit.py script will create plots of the receiver operating characteristic curves for each of the output nodes in the DNN.
- Plots contain the AUC as figure of merit.
- The plots can be found in the 'plots' directory within the directory the TMVA factory created during training.
!WARNING! - Currently doesnt work with LCG builds of python 2.7! Perform the following without enabling LCG filesystem via 'scl' command.

- Following on from the previous examples, run the command:
```
python DNN_ROCit.py -s <Suffix_of_training_directory>
```
For example, following on from the command used to train the network in the above section:
```
python DNN_ROCit.py -s 2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs
```
- This will create the ROC curve plots and place them in the directory '2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs/plots'.

## Monitoring Training Plots
- To obtain training monitoring as a function of the training epochs.
- Using Tensorboard (see argument in BookMethod) we get a log output file (put in ./logs directory).
- Create ssh tunnel loggin into lxplus machine where file was created e.g. for file:
```
MultiClass_DNN_2HLs_relu_D-VarTrans_0.008-learnRate_10-epochs/logs/events.out.tfevents.1524833056.lxplus094.cern.ch
```
one could ssh onto an lxplus machine:
```
ssh -D 8080 jthomasw@lxplus094.cern.ch
```
- Run command:
```
tensorboard --logdir <DNN_training_directory>/logs/ --port 8080
```
e.g.:
```
tensorboard --logdir MultiClass_DNN_2HLs_relu_D-VarTrans_0.008-learnRate_100-epochs/logs/ --port 8080
```
- Setup tunnel in browser and go to 'http://lxplus094.cern.ch:8080'

## Application of DNN weights
- To apply the weights obtained after training the DNN one can use the code 'apply_trained_DNN.py'.
- The code requires two arguments.
- The first is the same .json list of arguments used during training.
- The second is the suffix of the directory where the training weights were stored.
- The following example command should demonstrate the usage:
```
python apply_trained_DNN.py -j input_variables_list.json -s <Suffix_of_directory_containing_training_weights>
```
Following on from the examples before:
```
python apply_trained_DNN.py -j input_variables_list.json -s 2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs
```

## Application Plotter
- Much like the script DNN_ResponsePlotter.py we have a similar script to plot the response of the nodes after applying the network weights.
- It runs in the same way:
```
python DNN_ApplicationPlotter.py -s <Suffix_of_directory_containing_training_weights>
```
So following on from the preceding examples:
```
python DNN_ApplicationPlotter.py -s 2HLs_relu_D+G-VarTrans_0.008-learnRate_10-epochs
```
- The code will take the outputs from apply_trained_DNN.py and plot the histograms in those ttrees.

## Evaluation
- A sub-directory called 'Evaluation' exists that contains a script called evaluate_DNN.py which contains the code to evaluate the DNN and categorise events for a given ntuple, based on a training file defined in an input directory.
- To make things faster, 'Evaluation' also contains lxbatch scripts to run each job on an lxbatch node (e.g. lxbatch_runjob_evaluateDNN_tthsample.sh) and a submission script to submit all jobs to lxbatch (e.g.lxbatch_submit_evaluateDNN.sh).
- For an example of how to text locally simply look at the python command inside one of the 'runjob' scripts.
- If you want to run a new trained DNN model, it is advised to first copy the training directory from one level above to this sub-directory to here.
- Then one needs to change the training .xml file to ensure the path to the trained model .h5 file is the full path, not the relative path as default otherwise the code will abort on the lxbatch node.
- For example commands, one can check the lxbatch scripts.
- The binning optimised and used in this script should be used to make templates for the fit in the analysis.

## Comparison with BDT
- The script 'apply_trained_BDTG.py' was written to apply the 2017-2018 analysis' BDTG weights to the 2017 samples and create and output .root file that can be used to create performance plots for the BDTG for comparison.
- The code has one option '-s' that is the suffix you wish to use. The suffix must be either 'ttHvsttJets' or 'ttHvsttV' which represents the BDT weights you want to apply.
- One must ensure that the directory that contains your classifier weights has this suffix e.g. '<prefix_of_directory_name_>ttHvsttJets'
- Example run command:
```
python apply_trained_BDTG.py -s ttHvsttJets
```
