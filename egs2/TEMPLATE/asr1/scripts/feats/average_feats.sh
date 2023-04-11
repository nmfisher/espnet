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

split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/feats.${n}.scp"
done

feats=${data}/feats.scp

utils/split_scp.pl ${feats} ${split_scps} || exit 1;

durations=${data}/durations
text=${data}/text

for n in $(seq ${nj}); do
  ./utils/filter_scp.pl ${logdir}/feats.${n}.scp  ${durations}  > ${logdir}/durations.$n
  ./utils/filter_scp.pl ${logdir}/feats.${n}.scp  ${text} > ${logdir}/text.$n
  ./utils/filter_scp.pl ${logdir}/feats.${n}.scp  ${data}/phone_word_mappings > ${logdir}/phone_word_mappings.$n  
done

${cmd} JOB=1:${nj} ${logdir}/avg_bfcc_${name}.JOB.log pyscripts/feats/average-word.py ${logdir}/feats.JOB.scp ${logdir}/durations.JOB ${logdir}/text.JOB ark,scp:${bfccdir}/avg_bfcc_${name}.JOB.ark,${bfccdir}/avg_bfcc_${name}.JOB.scp ${logdir}/phone_word_mappings.JOB ark,scp:${logdir}/word_phone_mappings.JOB.ark,${logdir}/word_phone_mappings.JOB.scp 

# concatenate the .scp files together.
for n in $(seq ${nj}); do
    cat ${bfccdir}/avg_bfcc_${name}.$n.scp || exit 1;
done > ${data}/feats_word_avg.scp || exit 1

for n in $(seq ${nj}); do
    cat ${logdir}/word_phone_mappings.$n.scp || exit 1;
done > ${data}/word_phone_mappings.scp || exit 1

# rm -f ${logdir}/feats.*.scp ${logdir}/durations.* ${logdir}/pitch.* ${logdir}/text.* ${logdir}/word_phone_mappings.* 2>/dev/null

nf=$(wc -l < ${data}/feats_word_avg.scp)
nu=$(wc -l < ${data}/wav.scp)
if [ ${nf} -ne ${nu} ]; then
    echo "It seems not all of the feature files were successfully ($nf != $nu);"
    echo "consider using utils/fix_data_dir.sh $data"
fi

echo "Averaged $nf BFCC features for $name"
