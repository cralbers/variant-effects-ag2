#!/bin/bash
#SBATCH --account=pas2905
#SBATCH --partition=nextgen
#SBATCH --job-name=atlas
#SBATCH --cpus-per-task=10
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

# Run from the script directory so .env and default Parquet paths resolve
# on the compute node, regardless of where sbatch was launched.
cd "$(dirname "$(readlink -f "$0")")"

# Large checkpoints: uncomment and point at project space.
# Home (/users) fills up; scratch (/fs/scratch) is purged.
# export ATLAS_OUTPUT_DIR=/fs/ess/PAS2905/coraalbers/atlas_out

# load the necessary modules
module load miniconda3/24.1.2-py310

# activate the alphagenome conda environment
source activate py311


## run script below
python atlas.py