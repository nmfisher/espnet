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

if [ $# -lt 2 ] || [ $# -gt 14 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

train_wav=$1
train_durations=$2
train_transcript=$3
train_pitch_out=$4
train_energy_out=$5
valid_wav=$6
valid_durations=$7
valid_transcript=$8
valid_pitch_out=$9
valid_energy_out=${10}
f0min=${11}
f0max=${12}
train_dump_dir=${13}
valid_dump_dir=${14}

data=$(dirname ${train_durations})
logdir=${data}/log
mkdir -p ${logdir}

sr=16000
hop_length=160

# TODO - put sample rate/hop_len/num clusters etc into config
nj=1
echo "Extracting F0 using durations at ${train_durations}"
${cmd} JOB=1:${nj} ${logdir}/extract_f0.JOB.log \
    pyscripts/feats/extract-f0.py ${sr} ${hop_length}  \
    ${train_wav} ${train_durations} ${train_transcript}  ${train_pitch_out} ${train_energy_out} \
    ${valid_wav} ${valid_durations} ${valid_transcript} ${valid_pitch_out} ${valid_energy_out} \
    ${f0min} ${f0max} ${train_dump_dir} ${valid_dump_dir}

