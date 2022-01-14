#!/usr/bin/env bash

# Begin configuration section.
nj=4
cmd=run.pl
verbose=0
filetype=""
preprocess_conf=""
# End configuration section.

help_message=$(cat << EOF
Usage: $0 [options] <train-durations> <train-transcript> <valid-durations> <valid-transcript>  [<log-dir>]
e.g.: $0 teacher_train_dir/durations data/train/text teacher_valid_dir/durations data/valid/text data/train/log
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

if [ $# -lt 2 ] || [ $# -gt 6 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

train_wav=$1
train_durations=$2
train_transcript=$3
train_clusters_out=$4
valid_wav=$5
valid_durations=$6
valid_transcript=$7
valid_clusters_out=$8

data=$(dirname ${train_durations})
logdir=${data}/log
mkdir -p ${logdir}

# TODO - put sample rate/hop_len/num clusters etc into config
nj=1
${cmd} JOB=1:${nj} ${logdir}/cluster_durations.JOB.log \
    pyscripts/feats/cluster-f0.py 16000 256 15 --verbose ${verbose}  \
    ${train_wav} ${train_durations} ${train_transcript}  ${train_clusters_out} ${valid_wav} ${valid_durations} ${valid_transcript} ${valid_clusters_out}

