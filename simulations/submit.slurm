#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --array=1-24
#SBATCH --job-name=ab_dyn
#SBATCH -n 1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" params.txt`
parArray=($parameters)

f=${parArray[0]}
d=${parArray[1]}
da=5

# check which commit we are in, and prepend the short hash
# to the output directory
current_commit() {
	if $(git rev-parse --is-inside-work-tree >/dev/null 2>&1); then
		local str="$(git rev-parse --short HEAD)"
	else
		local str="no-git"
	fi
	echo "$str"
}
if [ $1 ]; then
	subdir=$1
else
	subdir=$(current_commit)
fi

outprefix=$subdir"/F="$f"_D="$d"_Dadh="$da

python neuraltube.py --prefix "$outprefix" --init "200." -t "200." --every "0.2" --dt "0.001" --diff-adh "$da" -f "$f" -D "$d" 2>&1 | tee outputs/"$outprefix".log

