for BSi in 6000
do
    for LRi in 0.00006
    do
	for EPi in 2
	do
	    for NNi in 4
	    do
		for LVi in 7
		do
		    for GPUN in 1
		    do
			sbatch --export=BS=${BSi},LR=${LRi},LV=${LVi},NN=${NNi},EPO=${EPi},GPU=${GPUN} --gres=gpu:v100:${GPUN} --output=/hpcfs/bes/mlgpu/kapoor/work/slc7/new/ttH_multilepton/CPStudies/outfiles/slurm_BS${BSi}_LR${LRi}_LV${LVi}_NN${NNi}_EPO${EPi}_GPU${GPUN}_%u_%x_%j.out slurm_gpu_Keras.sh
		    done
		done
	    done
	done
    done
done


# for BSi in 8112
# do
#     for LRi in 0.0001 0.00001
#     do
# 	for EPi in 2
# 	do
# 	    for NNi in 4 5 6
# 	    do
# 		for LVi in 5 6
# 		do
# 		    sbatch --export=BS=$BSi,LR=$LRi,LV=$LVi,NN=$NNi,EPO=$EPi slurm_sample_script_gpu_example_TMVA2.py
# 		done
# 	    done
# 	done
#     done
# done
