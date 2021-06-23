#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --array=1-24
#SBATCH --job-name=test_diff
#SBATCH -n 4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL


parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" params.txt`
parArray=($parameters)

f=${parArray[0]}
d=${parArray[1]}

outprefix=diff_F"$f"_D"$d"
python neuraltube.py --prefix "$outprefix" --init "100." -t "200." --every "0.2" --dt "0.002" -f "$f" -D "$d" --continue 2>&1 | tee outputs/diff_F"$f"_D"$d".log

