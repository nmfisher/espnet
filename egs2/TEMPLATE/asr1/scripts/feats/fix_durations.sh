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

if [ $# -lt 2 ] || [ $# -gt 7 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

train_feats=$1
valid_feats=$2
train_durations=$3
train_durations_out=$4
valid_durations=$5
valid_durations_out=$6
odim=$7

data=$(dirname ${train_durations_out})
logdir=${data}/log
mkdir -p ${logdir}

# TODO - put sample rate/hop_len/num clusters etc into config
nj=1
${cmd} JOB=1:${nj} ${logdir}/fix_durations.JOB.log \
    pyscripts/feats/fix-durations.py ${train_feats} ${valid_feats} ${train_durations} ${train_durations_out} ${valid_durations} ${valid_durations_out} $odim

if [ ! -f ${train_durations_out} ] || [ ! -f ${valid_durations_out} ]; then
    echo "Error fixing durations"
    exit -1;
fi
