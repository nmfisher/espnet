#!/usr/bin/env bash

# Begin configuration section.
nj=4
cmd=run.pl
verbose=0
filetype=""
preprocess_conf=""
# End configuration section.

help_message=$(cat << EOF
Usage: $0 [options] <data_dir> <f0min> <f0max> 
e.g.: $0 data/train 80 8000 
data_dir must contain:
- wav.scp
- text


Options:
  --nj <nj>                                        # number of parallel jobs
  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs.
  --verbose <num>                                  # Default: 0
EOF
)

echo "$0 $*" 1>&2 # Print the command line for logging

. parse_options.sh || exit 1;

if [ $# -lt 2 ] || [ $# -gt 8 ]; then
    echo "${help_message}" 1>&2
    exit 1;
fi

set -euo pipefail

dir=$1
f0min=${2}
f0max=${3}

wav=$dir/wav.scp
durations=$dir/durations
transcript=$dir/text
pitch_out=$dir/pitch
energy_out=$dir/energy

logdir=$dir/log
mkdir -p ${logdir}
dump_dir=$logdir

sr=16000
hop_length=160

# TODO - put sample rate/hop_len/num clusters etc into config
split_scps=""
for n in $(seq ${nj}); do
    split_scps="${split_scps} ${logdir}/wav.${n}.scp"
done

utils/split_scp.pl ${wav} ${split_scps} || exit 1;

for n in $(seq ${nj}); do
  ./utils/filter_scp.pl ${logdir}/wav.${n}.scp  ${durations}  > ${logdir}/durations.$n
  ./utils/filter_scp.pl ${logdir}/wav.${n}.scp  ${transcript} > ${logdir}/transcript.$n
done

echo "Extracting F0 using durations at ${durations}, pitch/energy will be written to ${pitch_out}.scp and ${energy_out}.scp"
${cmd} JOB=1:${nj} ${logdir}/extract_f0.JOB.log \
    pyscripts/feats/extract-f0.py ${sr} ${hop_length}  \
    ${logdir}/wav.JOB.scp ${logdir}/durations.JOB ${logdir}/transcript.JOB ${logdir}/pitch.JOB ${logdir}/energy.JOB \
    ${f0min} ${f0max} ${dump_dir}
echo "done!"
for n in $(seq ${nj}); do
    cat ${logdir}/pitch.$n.scp || exit 1;
done > ${pitch_out}.scp || exit 1

for n in $(seq ${nj}); do
    cat ${logdir}/energy.$n.scp || exit 1;
done > ${energy_out}.scp || exit 1

# rm -f ${logdir}/wav.*.scp ${logdir}/durations.* ${logdir}/transcript.* ${logdir}/pitch.*.scp ${logdir}/energy.*.scp
