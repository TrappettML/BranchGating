#!/bin/bash

#SBATCH --partition=computelong,computelong_intel
#SBATCH --job-name=DistSomaFunc
#SBATCH --output=Futures_Parallel/%x-%A-%a.out
#SBATCH --error=Futures_Parallel/%x-%A-%a.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G


#SBATCH --time=14-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --account=tau  ### Account used for job submission

#SBATCH --mail-type=all
#SBATCH --mail-user=mtrappet@uoregon.edu


source /home/mtrappet/BranchGating/data-science/bin/activate
params=$(sed -n "${SLURM_ARRAY_TASK_ID}p" params.txt)

python ../experiments/FuturesLongLearning.py $params