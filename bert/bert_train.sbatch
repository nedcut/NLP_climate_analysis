#!/usr/bin/env bash
# slurm template for serial jobs
# Set SLURM options
#SBATCH --job-name=Climate_Model
#SBATCH --output=Climate_Model_train-%j.out
# Standard output and error log
#SBATCH --mail-user=pkcutler@middlebury.edu
# Where to send mail
#SBATCH --mail-type=ALL
# Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=1gb
# Job memory request 
#SBATCH --partition=standard
# Partition (queue) 
#SBATCH --time=00:10:00
# Time limit hrs:min:sec 

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 
# Running your code here 
python train.py
# End of job info 
echo "Ending: "`date +"%D %T"`