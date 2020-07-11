source /hpcfs/bes/mlgpu/kapoor/anaconda3/etc/profile.d/conda.sh
export ANACONDA="/hpcfs/bes/mlgpu/kapoor/anaconda3/"
export PATH=$ANACONDA"bin/:$PATH"
export LD_LIBRARY_PATH=$ANACONDA"lib/:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$ANACONDA"pkgs/cudnn-7.3.1-cuda10.0_0/lib/:$LD_LIBRARY_PATH"
conda activate roottestenv
#source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc7-opt/setup.sh
