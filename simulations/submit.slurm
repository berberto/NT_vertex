#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --array=1-24
#SBATCH --job-name=test_diff
#SBATCH -n 4
#SBATCH --mem=128G
#SBATCH --time=0:01:00
#SBATCH --mail-type=BEGIN,END,FAIL


parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" params.txt`
parArray=($parameters)

f=${parArray[0]}
d=${parArray[1]}
da=5
outprefix="diff_F="$f"_D="$d"_Dadh="$da

python neuraltube.py --dry --prefix "$outprefix" --init "200." -t "200." --every "0.2" --dt "0.002" --diff-adh "$da" -f "$f" -D "$d" 2>&1 | tee outputs/"$outprefix".log

