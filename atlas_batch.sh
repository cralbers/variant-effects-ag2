#!/bin/bash
#SBATCH --account=pas2905
#SBATCH --partition=nextgen
#SBATCH --job-name=atlas
#SBATCH --cpus-per-task=10
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err


# load the necessary modules
module load miniconda3/24.1.2-py310

# activate the alphagenome conda environment
source activate py311


## run script below
python atlas.py