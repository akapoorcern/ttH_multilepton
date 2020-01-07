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
#SBATCH --job-name=test
  
# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=/hpcfs/bes/mlgpu/kapoor/work/slc7/ttH_multilepton/keras-DNN/slurm_%u_%x_%j.out
  
# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=30000
  
# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:2
    
######## Part 2 ######
# Script workload    #
######################
  
# Replace the following lines with your real workload
Fname='full_3000_newArchitecture2'
#Fname=$fname
echo $Fname$

###########################################
pyFname='train-DNN_'$Fname'.py'
dir='2017samples_'$Fname'_tH_InverseSRYields'
cd /hpcfs/bes/mlgpu/kapoor/work/slc7/ttH_multilepton/keras-DNN
source ./setup.sh
date
time(python $pyFname -t 1 -s tH)
tar -zcvf $Fname.tar.gz $(find ./$dir/ -type f '!' -name '*.csv')
##########################################
# list the allocated hosts
srun -l hostname
  
# list the GPU cards of the host
/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"
  
sleep 180 
