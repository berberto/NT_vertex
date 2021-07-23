#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --array=1-32
#SBATCH --job-name=ab_dyn
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

parameters=`sed -n "${SLURM_ARRAY_TASK_ID} p" params.txt`
parArray=($parameters)

k=${parArray[0]}
r=${parArray[1]}
e=${parArray[2]}
f=${parArray[3]}
d=${parArray[4]}
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

outprefix=$subdir"/K="$k"_r="$r"_eps="$e"_F="$f"_D="$d

python neuraltube.py --prefix "$outprefix" \
	--init "200." -t "200." --every "0.2" --dt "0.001" \
	--nucl-stiff "$k" --nucl-crowd "$r" --nucl-noise "$e" -f "$f" -D "$d" \
	--diff-adh "$da" 2>&1 | tee outputs/"$outprefix".log

