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

if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

wav=$1
out=$2
utt2spk=$3
stats=$4
create_stats=
if [ $# -eq 5 ]; then
  create_stats=$5
fi

data=$(dirname ${wav})
logdir=${data}/log
mkdir -p ${logdir}

sr=16000
hop_length=160

# TODO - put sample rate/hop_len/num clusters etc into config
nj=1
echo "Extracting speaker embeddings"
_opts="$wav $out $utt2spk"

if [ -z "$create_stats" ]; then
  _opts+=" --stats $stats"
else
  _opts+=" --stats_out $stats"
fi

echo "Using spkembs opts $_opts"

PYTHONPATH=/home/hydroxide/projects/byol-a ${cmd} JOB=1:${nj} ${logdir}/extract_spkembs.JOB.log \
    python3.7 pyscripts/feats/extract-spkembs.py $_opts

