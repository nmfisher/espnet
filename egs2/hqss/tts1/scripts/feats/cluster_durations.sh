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

if [ $# -lt 2 ] || [ $# -gt 7 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

train_durations=$1
train_transcript=$2
train_clusters_out=$3
valid_durations=$4
valid_transcript=$5
valid_clusters_out=$6
num_clusters=$7

data=$(dirname ${train_durations})
logdir=${data}/log
mkdir -p ${logdir}

# TODO - put sample rate/hop_len/num clusters etc into config
nj=1
${cmd} JOB=1:${nj} ${logdir}/cluster_durations.JOB.log \
    pyscripts/feats/cluster-durations.py --num_clusters $num_clusters --verbose ${verbose}  \
    --train_durations ${train_durations} \
    --train_transcripts ${train_transcript} \
    --train_outfile ${train_clusters_out} \
    --valid_durations ${valid_durations}  \
    --valid_transcripts ${valid_transcript} \
    --valid_outfile ${valid_clusters_out}
echo "DONE!"
