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

if [ $# -lt 5 ] || [ $# -gt 5 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

train_wav=$1
test_wav=$2
train_out=$3
test_out=$4
stats=$5

data=$(dirname ${train_wav})
logdir=${data}/log
mkdir -p ${logdir}

sr=16000
hop_length=160

# TODO - put sample rate/hop_len/num clusters etc into config
nj=1
echo "Extracting speaker embeddings"
_opts="$train_wav $test_wav $train_out $test_out $stats"

echo "Using spkembs opts $_opts"

PYTHONPATH=/home/hydroxide/projects/byol-a ${cmd} JOB=1:${nj} ${logdir}/extract_spkembs.JOB.log \
    python3.7 pyscripts/feats/extract-spkembs.py $_opts

