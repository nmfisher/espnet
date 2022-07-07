#!/usr/bin/env bash

# Begin configuration section.
nj=4
cmd=run.pl
verbose=0
filetype=""
preprocess_conf=""
# End configuration section.

help_message=$(cat << EOF
Usage: $0 [options] <wav> 
e.g.: $0 data/train/wav.scp 

Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --verbose <num>                                  # Default: 0
EOF
)

echo "$0 $*" 1>&2 # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 1 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

wav=$1

data=$(dirname ${wav})
logdir=${data}/log
mkdir -p ${logdir}

sr=16000
hop_length=160

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
done

utils/split_scp.pl ${wav} ${split_scps} || exit 1;

echo "Extracting speaker embeddings"

#PYTHONPATH=/home/hydroxide/projects/byol-a 
${cmd} JOB=1:${nj} ${logdir}/extract_spkembs.JOB.log \
    python3.7 pyscripts/feats/extract-spkembs.py scp:${logdir}/wav.JOB.scp ark,scp:${logdir}/xvector.JOB.ark,${logdir}/xvector.JOB.scp || exit 1;

# "ark,scp:${data_feats}${_suf}/${valid_set}/xvector.ark,${data_feats}${_suf}/${valid_set}/xvector.scp"                 

for n in $(seq ${nj}); do
  cat ${logdir}/xvector.${n}.scp || exit 1;
done > ${data}/xvector.scp || exit 1

rm -f ${logdir}/xvector.*.scp 

nf=$(wc -l < ${data}/xvector.scp)
nu=$(wc -l < ${data}/wav.scp)
if [ ${nf} -ne ${nu} ]; then
    echo "It seems not all speaker embeddings were successful ($nf != $nu);"
    echo "consider using utils/fix_data_dir.sh $data"
fi