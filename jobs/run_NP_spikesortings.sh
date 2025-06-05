#!/bin/bash -l

# Request 1 hour of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=48:00:00
 
# For 1 GPU
#$ -l gpu=2

# Request 256 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=32G

# Request 10 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=200G

# Set the name of the job.
#$ -N spikesorting_NP

# Request 6 cores.
#$ -pe smp 6

cd $TMPDIR

# Set the working directory to somewhere in your scratch space.  
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucjuhae/Scratch/workspace
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/10.1.243/gnu-4.9.2
module load cudnn/7.6.5.32/cuda-10.1
# module load xorg-utils/X11R7.7

nvidia-smi

# Activate python environment
conda activate spikeenv

# Activate python environment
#source /home/ucjuhae/myenv/bin/activate

# Your work should be done in $TMPDIR 


python /home/ucjuhae/spikesorting_scripts/scripts/spikesorting_concatenated_NP.py /home/ucjuhae/spikesorting_scripts/scripts/json_files/spikesorting_params_concatenated_NP.json