#!/usr/bin/env bash

# Begin configuration section.
nj=4
cmd=run.pl
verbose=0
filetype=""
preprocess_conf=""
# End configuration section.

help_message=$(cat << EOF
Usage: $0 [options] <train-wav> <train-durations> <train-transcript> <valid-wav> <valid-durations> <valid-transcript> <f0min> <f0max>  [<log-dir>]
e.g.: $0 data/train/wav.scp teacher_train_dir/durations data/train/text data/test/wav.scp teacher_valid_dir/durations data/valid/text data/train/log
assuming data/train/text contains the phonetic transcript like:
SPKR1_UTT1 h eh l l o

Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --verbose <num>                                  # Default: 0
EOF
)

echo "$0 $*" 1>&2 # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 2 ] || [ $# -gt 11 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

train=$1
valid=$2
num_clusters=$3

data=$(dirname ${train})
logdir=${data}/log
mkdir -p ${logdir}

sr=16000
hop_length=256

echo "Clustering F0 and energy in ${train} and ${valid}, output to ${train}/pitch_clusters and ${valid}/pitch_clusters"

# TODO - put sample rate/hop_len/num clusters etc into config
nj=1
${cmd} JOB=1:${nj} ${logdir}/cluster_f0.JOB.log \
    pyscripts/feats/cluster-f0.py --num_clusters ${num_clusters} \
    ${train}/text \
    ${train}/pitch.scp \
    ${train}/energy.scp \
    ${train}/pitch_clusters \
    ${train}/energy_clusters \
    ${valid}/text  \
    ${valid}/pitch.scp  \
    ${valid}/energy.scp \
    ${valid}/pitch_clusters \
    ${valid}/energy_clusters 

