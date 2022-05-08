#!/usr/bin/env bash

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Begin configuration section.
nj=1
cmd=run.pl
# End configuration section.

help_message=$(cat <<EOF
Usage: $0 [options] <train_wav.scp> <valid_wav.scp>
e.g.: $0 data/train/wav.scp data/valid/wav.scp
Options:
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
EOF
)
echo "$0 $*"  # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

sample_rate=16000

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=${data}/log
fi
if [ $# -ge 3 ]; then
  bfccdir=$3
else
  bfccdir=${data}/data
fi

# make $bfccdir an absolute pathname.
bfccdir=$(perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${bfccdir} ${PWD})

# use "name" as part of name of the archive.
name=$(basename ${data})

mkdir -p ${bfccdir} || exit 1;
mkdir -p ${logdir} || exit 1;

if [ -f ${data}/feats.scp ]; then
  mkdir -p ${data}/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  cp ${data}/feats.scp ${data}/.backup
fi

feats=${data}/feats.scp
durations=${data}/durations
text=${data}/text

${cmd} JOB=1:${nj} ${logdir}/avg_bfcc_${name}.JOB.log \
    pyscripts/feats/average-word.py ${feats} ${durations} ${text} ark,scp:${bfccdir}/avg_bfcc_${name}.ark,${bfccdir}/avg_bfcc_${name}.scp ark,scp:${data}/phone_word_mappings.ark,${data}/phone_word_mappings.scp  ark,scp:${data}/word_phone_mappings.ark,${data}/word_phone_mappings.scp 

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat ${bfccdir}/avg_bfcc_${name}.scp || exit 1;
done > ${data}/feats_word_avg.scp || exit 1

rm -f ${logdir}/wav.*.scp ${logdir}/segments.* 2>/dev/null

nf=$(wc -l < ${data}/feats_word_avg.scp)
nu=$(wc -l < ${data}/wav.scp)
if [ ${nf} -ne ${nu} ]; then
    echo "It seems not all of the feature files were successfully ($nf != $nu);"
    echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Succeeded averaging BFCC features for $name"
