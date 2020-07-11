#! /bin/bash
  
######## Part 1 #########
# Script parameters     #
#########################
 
# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu
 
# Specify the QOS, mandatory option
#SBATCH --qos=normal
 
# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
# write the experiment which will pay for your resource consumption
#SBATCH --account=mlgpu
  
# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=Unti
#TMVA_Higgs_Classification_Study6_NewSamples
#TMVA_Higgs_Classification_April_BDThyperparameter
#_modi2_newJSON
#newDNNs_BiggerBatch3
#newDNNs_ExactJoshuha
##newDNNs_13thJan_JoshuhaExact_vs_JoshuaHigherBatch
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
###SBATCH --output=/hpcfs/bes/mlgpu/kapoor/work/slc7/ttH_multilepton/keras-DNN/CPStudies/outfiles/slurm_%u_%x_%j.out
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=30000
  
# Specify how many GPU cards to use
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
########################################
#Fname='py3_Joshuha_BS_XMAS'
#_modi2_newJSON
#'newDNNs_13thJan_JoshuhaExact_vs_JoshuaHigherBatch'
#Fname=$fname
#echo $Fname$
#pyFname='train-DNN_'$Fname'.py'
#dir='2017samples_full_'$Fname'_tH_InverseSRYields'
cd /hpcfs/bes/mlgpu/kapoor/work/slc7/new/ttH_multilepton/CPStudies/
#source /cvmfs/sft.cern.ch/lcg/views/dev3python3/latest/x86_64-centos7-gcc8-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh
source setup.sh
date
#time(python $pyFname -t 1 -s tH)
#time(python Untitled123.py)
#time(python Untitled1234.py)
#time(python CPStudy_KerasDNN_withAdj_v1_MGPU.py $BS $LR $EPO $LV $NN $GPU)
time(python CPStudy_KerasDNN_withAdj_v1.py $BS $LR $EPO $LV $NN $GPU)
#tar -zcvf $dir-Folder.tar.gz $(find ./$dir/ -type f '!' -name '*.csv')
##########################################
# Work load end

# Do not remove below this line

# list the allocated hosts
srun -l hostname
  
# list the GPU cards of the host
/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"
  
sleep 180 
