#!/usr/bin/env bash

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
skip_upload_hf=true # Skip uploading to hugging face stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=8                # The number of parallel jobs.
inference_nj=8      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts="" # Options to be passed to local/data.sh.

# Feature extraction related
feats_type=lpcnet             # Input feature type.
audio_format=wav          # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
min_wav_duration=0.1       # Minimum duration in second.
max_wav_duration=20        # Maximum duration in second.
use_xvector=false          # Whether to use x-vector (Require Kaldi).
use_sid=false              # Whether to use speaker id as the inputs (Need utt2spk in data directory).
use_lid=false              # Whether to use language id as the inputs (Need utt2lang in data directory).
feats_extract=None        # On-the-fly feature extractor.
feats_normalize=None # On-the-fly feature normalizer.
fs=16000                   # Sampling rate.
fmin=80                    # Minimum frequency of Mel basis.
fmax=7600                  # Maximum frequency of Mel basis.
n_mels=80                  # The number of mel basis.
# Only used for the model using pitch & energy features (e.g. FastSpeech2)
f0min=80  # Maximum f0 for pitch extraction.
f0max=400 # Minimum f0 for pitch extraction.


# Vocabulary related
oov="<unk>"         # Out of vocabrary symbol.
blank="<blank>"     # CTC blank symbol.
sos_eos="<sos/eos>" # sos and eos symbols.

# Training related
train_config=""    # Config for training.
student_train_config="" # Config for student model
train_args=""      # Arguments for training, e.g., "--max_epoch 1".
                   # Note that it will overwrite args in train config.
tag=""             # Suffix for training directory.
tts_exp=""         # Specify the directory path for experiment. If this option is specified, tag is ignored.
tts_stats_dir=""   # Specify the directory path for statistics. If empty, automatically decided.
num_splits=1       # Number of splitting for tts corpus.
teacher_dumpdir="" # Directory of teacher outputs (needed if tts=fastspeech).
write_collected_feats=false # Whether to dump features in stats collection.
tts_task=tts                # TTS task (tts or gan_tts).

# Decoding related
inference_config="" # Config for decoding.
inference_args=""   # Arguments for decoding (e.g., "--threshold 0.75").
                    # Note that it will overwrite args in inference config.
inference_tag=""    # Suffix for decoding directory.
inference_model=train.loss.ave.pth # Model path for decoding.
                                   # e.g.
                                   # inference_model=train.loss.best.pth
                                   # inference_model=3epoch.pth
                                   # inference_model=valid.acc.best.pth
                                   # inference_model=valid.loss.ave.pth
vocoder_file=none  # Vocoder parameter file, If set to none, Griffin-Lim will be used.
download_model=""  # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=""     # Name of training set.
valid_set=""     # Name of validation set used for monitoring/tuning network training.
test_sets=""     # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
srctexts=""      # Texts to create token list. Multiple items can be specified.
nlsyms_txt=none  # Non-linguistic symbol list (needed if existing).
token_type=phn   # Transcription type (char or phn).
cleaner=tacotron # Text cleaner.
g2p=g2p_en       # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.

# Upload model related
hf_repo=

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>" --srctexts "<srctexts>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data prep related
    --local_data_opts # Options to be passed to local/data.sh (default="${local_data_opts}").

    # Feature extraction related
    --feats_type       # Feature type (default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").
    --use_xvector      # Whether to use X-vector (Require Kaldi, default="${use_xvector}").
    --use_sid          # Whether to use speaker id as the inputs (default="${use_sid}").
    --use_lid          # Whether to use language id as the inputs (default="${use_lid}").
    --feats_extract    # On the fly feature extractor (default="${feats_extract}").
    --feats_normalize  # Feature normalizer for on the fly feature extractor (default="${feats_normalize}")
    --fs               # Sampling rate (default="${fs}").
    --fmax             # Maximum frequency of Mel basis (default="${fmax}").
    --fmin             # Minimum frequency of Mel basis (default="${fmin}").
    --f0min            # Maximum f0 for pitch extraction (default="${f0min}").
    --f0max            # Minimum f0 for pitch extraction (default="${f0max}").
    --oov              # Out of vocabrary symbol (default="${oov}").
    --blank            # CTC blank symbol (default="${blank}").
    --sos_eos          # sos and eos symbole (default="${sos_eos}").

    # Training related
    --train_config  # Config for training (default="${train_config}").
    --student_train_config  # Config for training (default="${student_train_config}").
    --train_args    # Arguments for training (default="${train_args}").
                    # e.g., --train_args "--max_epoch 1"
                    # Note that it will overwrite args in train config.
    --tag           # Suffix for training directory (default="${tag}").
    --tts_exp       # Specify the directory path for experiment.
                    # If this option is specified, tag is ignored (default="${tts_exp}").
    --tts_stats_dir # Specify the directory path for statistics.
                    # If empty, automatically decided (default="${tts_stats_dir}").
    --num_splits    # Number of splitting for tts corpus (default="${num_splits}").
    --teacher_dumpdir       # Directory of teacher outputs (needed if tts=fastspeech, default="${teacher_dumpdir}").
    --write_collected_feats # Whether to dump features in statistics collection (default="${write_collected_feats}").
    --tts_task              # TTS task {tts or gan_tts} (default="${tts_task}").

    # Decoding related
    --inference_config  # Config for decoding (default="${inference_config}").
    --inference_args    # Arguments for decoding, (default="${inference_args}").
                        # e.g., --inference_args "--threshold 0.75"
                        # Note that it will overwrite args in inference config.
    --inference_tag     # Suffix for decoding directory (default="${inference_tag}").
    --inference_model   # Model path for decoding (default=${inference_model}).
    --vocoder_file      # Vocoder paramemter file (default=${vocoder_file}).
                        # If set to none, Griffin-Lim vocoder will be used.
    --download_model    # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh.
    --train_set          # Name of training set (required).
    --valid_set          # Name of validation set used for monitoring/tuning network training (required).
    --test_sets          # Names of test sets (required).
                         # Note that multiple items (e.g., both dev and eval sets) can be specified.
    --srctexts           # Texts to create token list (required).
                         # Note that multiple items can be specified.
    --nlsyms_txt         # Non-linguistic symbol list (default="${nlsyms_txt}").
    --token_type         # Transcription type (default="${token_type}").
    --cleaner            # Text cleaner (default="${cleaner}").
    --g2p                # g2p method (default="${g2p}").
    --lang               # The language type of corpus (default="${lang}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# Check feature type
if [ "${feats_type}" = "raw" ]; then
    data_feats="${dumpdir}/raw"
elif [ "${feats_type}" = lyra ]; then
    data_feats="${dumpdir}/raw"      
    odim=30
    feats_file="feats.scp"
    feats_filetype="kaldi_ark"
elif [ "${feats_type}" = "lpcnet" ]; then
    odim=20
    feats_file="feats.scp"
    feats_filetype="kaldi_ark"
    data_feats="${dumpdir}/raw"    
elif [ "${feats_type}" = "wlsc" ]; then
    data_feats="${dumpdir}/raw"    
elif [ "${feats_type}" = "encodec" ]; then
    data_feats="${dumpdir}/raw"
    odim=2    
    feats_file="feats.scp"
    feats_filetype="kaldi_ark"
else
    log "${help_message}"
    log "Error: only supported: --feats_type raw"
    exit 2
fi

# the original tts.sh script tokenized a source text to generate a list of tokens/token IDs
# we don't do this because:
# a) the list of tokens should remain constant (and even though it shouldn't ever happen, we don't want to accidentally omit a token during training because it wasn't in the source text)
# b) we want control over the exact token<->ID mappings (e.g. if we want to use exactly the same mappings in some other model
# this script expects the file data/symbol_ids.txt to exist, and copies to ${dump_dir/token_list/tokens.txt} prepending <blank> and appending <sos/eos>
# symbol_ids.txt has one token per line, where the line number (zero-indexed) represents the symbol ID.
src_tokens=$(cat data/train/text | cut -d' ' -f2- | tr ' ' '\n' | tr '[:upper:]' '[:lower:]' | sort | uniq )

mkdir -p "${dumpdir}/token_list"
token_list="${dumpdir}/token_list/tokens.txt"
rm -f $token_list
echo $blank > $token_list
echo "$src_tokens" >> $token_list
echo $oov >> $token_list
echo $sos_eos >> $token_list
_nj=$nj

# Set tag for naming of model directory
if [ -z "${tag}" ]; then
    if [ -n "${train_config}" ]; then
        tag="$(basename "${train_config}" .yaml)_${feats_type}_${token_type}"
    else
        tag="train_${feats_type}_${token_type}"
    fi
    if [ "${cleaner}" != none ]; then
        tag+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        tag+="_${g2p}"
    fi
    # Add overwritten arg's info
    if [ -n "${train_args}" ]; then
        tag+="$(echo "${train_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    inference_tag+="_$(echo "${inference_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# The directory used for collect-stats mode
if [ -z "${tts_stats_dir}" ]; then
    tts_stats_dir="${expdir}/tts_stats_${feats_type}"
    if [ "${feats_extract}" != fbank ]; then
        tts_stats_dir+="_${feats_extract}"
    fi
    tts_stats_dir+="_${token_type}"
    if [ "${cleaner}" != none ]; then
        tts_stats_dir+="_${cleaner}"
    fi
    if [ "${token_type}" = phn ]; then
        tts_stats_dir+="_${g2p}"
    fi
fi
# The directory used for training commands
if [ -z "${tts_exp}" ]; then
    tts_exp="${expdir}/tts_${tag}"
    student_tts_exp="${expdir}/tts_${tag}_student"
fi


# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi


    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        # TODO(kamo): Change kaldi-ark to npy or HDF5?
        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        # log "Stage 2: Format wav.scp: data/ -> ${data_feats}/"
        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"
            rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
            _opts=
            if [ -e data/"${dset}"/segments ]; then
                _opts+="--segments data/${dset}/segments "
            fi
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                "data/${dset}/wav.scp" "${data_feats}${_suf}/${dset}"
            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"
        done

        # Extract X-vector
        if "${use_xvector}"; then
            log "Stage 2+: Extract X-vector: data/ -> ${dumpdir}/xvector (Require Kaldi)"
            mkdir -p ${dumpdir}/xvector/${train_set} ${dumpdir}/xvector/${valid_set}
            ./scripts/feats/extract_spkembs.sh --nj ${_nj} ${data_feats}${_suf}/${train_set}/wav.scp
            ./scripts/feats/extract_spkembs.sh --nj ${_nj} ${data_feats}${_suf}/${valid_set}/wav.scp 
        fi

        # Prepare spk id input
        if "${use_sid}"; then
            log "Stage 2+: Prepare speaker id: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                if [ "${dset}" = "${train_set}" ]; then
                    # Make spk2sid
                    # NOTE(kan-bayashi): 0 is reserved for unknown speakers
                    echo "<unk> 0" > "${data_feats}${_suf}/${dset}/spk2sid"
                    cut -f 2 -d " " "${data_feats}${_suf}/${dset}/utt2spk" | sort | uniq | \
                        awk '{print $1 " " NR}' >> "${data_feats}${_suf}/${dset}/spk2sid"
                fi
                pyscripts/utils/utt2spk_to_utt2sid.py \
                    "${data_feats}/org/${train_set}/spk2sid" \
                    "${data_feats}${_suf}/${dset}/utt2spk" \
                    > "${data_feats}${_suf}/${dset}/utt2sid"
            done
        fi

        # Prepare lang id input
        if "${use_lid}"; then
            log "Stage 2+: Prepare lang id: data/ -> ${data_feats}/"
            for dset in "${train_set}" "${valid_set}" ${test_sets}; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                if [ "${dset}" = "${train_set}" ]; then
                    # Make lang2lid
                    # NOTE(kan-bayashi): 0 is reserved for unknown languages
                    echo "<unk> 0" > "${data_feats}${_suf}/${dset}/lang2lid"
                    cut -f 2 -d " " "${data_feats}${_suf}/${dset}/utt2lang" | sort | uniq | \
                        awk '{print $1 " " NR}' >> "${data_feats}${_suf}/${dset}/lang2lid"
                fi
                # NOTE(kan-bayashi): We can reuse the same script for making utt2sid
                pyscripts/utils/utt2spk_to_utt2sid.py \
                    "${data_feats}/org/${train_set}/lang2lid" \
                    "${data_feats}${_suf}/${dset}/utt2lang" \
                    > "${data_feats}${_suf}/${dset}/utt2lid"
            done
        fi
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            if [ -e "${data_feats}/org/${dset}/utt2sid" ]; then
                cp "${data_feats}/org/${dset}/utt2sid" "${data_feats}/${dset}/utt2sid"
            fi
            if [ -e "${data_feats}/org/${dset}/utt2lid" ]; then
                cp "${data_feats}/org/${dset}/utt2lid" "${data_feats}/${dset}/utt2lid"
            fi

            # Remove short utterances
            _fs=$($python -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$($python -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$($python -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            <"${data_feats}/org/${dset}/wav.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/wav.scp"

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            _fix_opts=""
            if [ -e "${data_feats}/org/${dset}/utt2sid" ]; then
                _fix_opts="--utt_extra_files utt2sid "
            fi
            if [ -e "${data_feats}/org/${dset}/utt2lid" ]; then
                _fix_opts="--utt_extra_files utt2lid "
            fi
            # shellcheck disable=SC2086

            utils/fix_data_dir.sh ${_fix_opts} "${data_feats}/${dset}"

            # Filter x-vector
            if "${use_xvector}"; then
                cp "${data_feats}${_suf}/${dset}"/xvector.{scp,scp.bak}
                <"${data_feats}${_suf}/${dset}/xvector.scp.bak" \
                    utils/filter_scp.pl "${data_feats}/${dset}/wav.scp"  \
                    >"${data_feats}/${dset}/xvector.scp"
            fi
        done
    fi
else
    log "Skip the stages for data preparation"
fi
# ========================== Data preparation is done here. ==========================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

    if [ $feats_type == "lpcnet" ]; then
        feat_script="make_lpcnet_feats.sh";
    elif [ $feats_type == "lyra" ]; then
        feat_script="make_lyra_feats.sh";
    elif [ $feats_type == "encodec" ]; then
        feat_script="make_encodec_feats.sh";
    fi
    ./scripts/feats/$feat_script --nj "${_nj}" ${data_feats}/${train_set} || exit -1;
    ./scripts/feats/$feat_script --nj "${_nj}" ${data_feats}/${valid_set} || exit -1;
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _teacher_train_dir="${teacher_dumpdir}/${train_set}"
        _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"

        ./utils/filter_scp.pl "${data_feats}/${train_set}/wav.scp" ${_teacher_train_dir}/durations > ${data_feats}/${train_set}/durations
        ./utils/filter_scp.pl "${data_feats}/${valid_set}/wav.scp" ${_teacher_valid_dir}/durations > ${data_feats}/${valid_set}/durations

        # LPCNet uses hop_length=160, sample rate = 16000
        # so frame length is 10ms, effective framerate = 100Hz
        if [ $feats_type == "lpcnet" ]; then
            factor=100
        elif [ $feats_type == "lyra" ]; then
        # Lyra 50Hz features
            factor=50
        elif [ $feats_type == "encodec" ]; then
        # Encodec 24khz model is 75Hz
            factor=150 # 75
        fi

        python ./pyscripts/feats/convert_second_to_frame_alignments.py ark,t:${data_feats}/${train_set}/durations scp:${data_feats}/${train_set}/feats.scp scp,ark,t:${data_feats}/${train_set}/durations.scp,${data_feats}/${train_set}/durations.ark $factor 
        python ./pyscripts/feats/convert_second_to_frame_alignments.py ark,t:${data_feats}/${valid_set}/durations scp:${data_feats}/${valid_set}/feats.scp scp,ark,t:${data_feats}/${valid_set}/durations.scp,${data_feats}/${valid_set}/durations.ark $factor 
        
        ./utils/fix_data_dir.sh "${data_feats}/${train_set}"
        ./utils/fix_data_dir.sh "${data_feats}/${valid_set}"
fi

# if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # for dset in "${train_set}" "${valid_set}"; do
    #     ./scripts/feats/extract_f0.sh --nj ${nj} ${data_feats}/${dset} ${f0min} ${f0max} 
    # done
# fi

num_prosody_clusters=$(grep "num_prosody_clusters" ${train_config} | sed "s/[^0-9+]//g");

#if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    #./scripts/feats/cluster_f0.sh ${data_feats}/${train_set} ${data_feats}/${valid_set} ${num_prosody_clusters}
    # ./scripts/normalize_f0.sh ${data_feats}/${train_set} ${data_feats}/${valid_set} 
#fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ] && [ ${feats_type} == "lpcnet" ]; then
    log "Stage 7: Averaging BFCCs per word"
    ./utils/filter_scp.pl ${data_feats}/${train_set}/${feats_file} data/${train_set}/phone_word_mappings > ${data_feats}/${train_set}/phone_word_mappings
    ./utils/filter_scp.pl ${data_feats}/${valid_set}/${feats_file} data/${valid_set}/phone_word_mappings > ${data_feats}/${valid_set}/phone_word_mappings
    ./scripts/feats/average_feats.sh --nj "${_nj}" ${data_feats}/${train_set}
    ./scripts/feats/average_feats.sh --nj "${_nj}" ${data_feats}/${valid_set}
fi

if ! "${skip_train}"; then
    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 8: TTS collect stats: train_set=${_train_dir}, valid_set=${_valid_dir}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

        _scp=$feats_file
        _type=$feats_filetype

        # _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/durations,durations,kaldi_ark "
        # _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/durations,durations,kaldi_ark "
        # _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/pitch_clusters.scp,pitch,kaldi_ark "
        # _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/pitch_clusters.scp,pitch,kaldi_ark "
        # _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/energy_clusters,energy,kaldi_ark "
        # _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/energy_clusters,energy,kaldi_ark "

        if "${use_xvector}"; then
            _xvector_train_dir="${data_feats}/${train_set}"
            _xvector_valid_dir="${data_feats}/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/xvector.scp,spembs,kaldi_ark "
            _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/xvector.scp,spembs,kaldi_ark "
        fi

        if "${use_sid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2sid,sids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2sid,sids,text_int "
            _opts+="--num_speakers $(cat "${data_feats}/org/${train_set}/spk2sid" | wc -l)"
        fi

        if "${use_lid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2lid,lids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2lid,lids,text_int "
        fi

        # 1. Split the key file
        _logdir="${tts_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_train_dir}/${_scp} wc -l)" "$(<${_valid_dir}/${_scp} wc -l)")

        key_file="${_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${tts_stats_dir}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${tts_stats_dir}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${tts_stats_dir}/run.sh"; chmod +x "${tts_stats_dir}/run.sh"

        # 3. Submit jobs
        log "TTS collect_stats started... log: '${_logdir}/stats.*.log'"

        log "LAUNCHING with _opts $_opts"

        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m "espnet2.bin.${tts_task}_train" \
                --collect_stats true \
                --write_collected_feats "${write_collected_feats}" \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                 --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --normalize none \
                --pitch_normalize none \
                --energy_normalize none \
                --train_data_path_and_name_and_type "${_train_dir}/text,text,text" \
                --train_data_path_and_name_and_type "${_train_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_valid_dir}/text,text,text" \
                --valid_data_path_and_name_and_type "${_valid_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_train_dir}/durations.scp,durations,kaldi_ark " \
                --valid_data_path_and_name_and_type "${_valid_dir}/durations.scp,durations,kaldi_ark " \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                --allow_variable_data_keys true \
                --odim "$odim " \
                ${_opts} ${train_args} || { cat "${_logdir}"/stats.1.log; exit 1; }
#                --train_data_path_and_name_and_type "${_train_dir}/pitch_clusters.scp,pitch,kaldi_ark   " \
#                --train_data_path_and_name_and_type "${_train_dir}/energy_clusters.scp,energy,kaldi_ark   " \
#                --valid_data_path_and_name_and_type "${_valid_dir}/pitch_clusters.scp,pitch,kaldi_ark   " \
#                --valid_data_path_and_name_and_type "${_valid_dir}/energy_clusters.scp,energy,kaldi_ark   " \

                #                 --train_data_path_and_name_and_type "${_train_dir}/word_phone_mappings.scp,word_phone_mappings,kaldi_ark  " \
                # --valid_data_path_and_name_and_type "${_valid_dir}/word_phone_mappings.scp,word_phone_mappings,kaldi_ark  " \
#                --train_data_path_and_name_and_type "${_train_dir}/phone_word_mappings,phone_word_mappings,text_int   " \
 #               --valid_data_path_and_name_and_type "${_valid_dir}/phone_word_mappings,phone_word_mappings,text_int   " \
        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${tts_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${tts_stats_dir}/train/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${tts_stats_dir}/train/text_shape.${token_type}"

        <"${tts_stats_dir}/valid/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${tts_stats_dir}/valid/text_shape.${token_type}"
    fi

    
    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        _train_dir="${data_feats}/${train_set}"
        _valid_dir="${data_feats}/${valid_set}"
        log "Stage 9: TTS Training: train_set=${_train_dir}, valid_set=${_valid_dir}, config=${train_config}"

        _opts=
        if [ -n "${train_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.tts_train --print_config --optim adam
            _opts+="--config ${train_config} "
        fi

         if "${use_xvector}"; then
            _xvector_train_dir="${data_feats}/${train_set}"
            _xvector_valid_dir="${data_feats}/${valid_set}"
            _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/xvector.scp,spembs,kaldi_ark "
            _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/xvector.scp,spembs,kaldi_ark "
        fi

        _teacher_train_dir="${teacher_dumpdir}/${train_set}"
        _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"
        
        _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
        _opts+="--train_shape_file ${tts_stats_dir}/train/text_shape.${token_type} "
        _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
        _opts+="--valid_shape_file ${tts_stats_dir}/valid/text_shape.${token_type} "

        _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/${feats_file},speech,${feats_filetype} "
        _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/${feats_file},speech,${feats_filetype} "

        _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/durations.scp,durations,kaldi_ark "
        _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/durations.scp,durations,kaldi_ark "
        # _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/pitch,pitch,npy "
        # _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/pitch,pitch,npy "
        # _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/energy,energy,npy "
        # _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/energy,energy,npy "
#        _opts+="--train_data_path_and_name_and_type ${_train_dir}/pitch_clusters.scp,pitch,kaldi_ark   "
#        _opts+="--train_data_path_and_name_and_type ${_train_dir}/energy_clusters.scp,energy,kaldi_ark   "
#        _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/pitch_clusters.scp,pitch,kaldi_ark   " 
#        _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/energy_clusters.scp,energy,kaldi_ark   "
#        _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/feats_word_avg.scp,feats_word_avg,kaldi_ark  " \
#        _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/feats_word_avg.scp,feats_word_avg,kaldi_ark  " \
#        _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/phone_word_mappings,phone_word_mappings,text_int   " \
#        _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/phone_word_mappings,phone_word_mappings,text_int   " \
        _opts+="--odim ${odim} "
        # _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/word_phone_mappings.scp,word_phone_mappings,kaldi_ark  " \
        # _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/word_phone_mappings.scp,word_phone_mappings,kaldi_ark  " \
        

        if [ -e ${_teacher_train_dir}/probs ]; then
            # Knowledge distillation case: use the outputs of the teacher model as the target
            _scp=$feats_file
            _type=npy
        else
            # Teacher forcing case: use groundtruth as the target
            _scp=wav.scp
            _type=sound
        fi


        # Add speaker ID to the inputs if needed
        if "${use_sid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2sid,sids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2sid,sids,text_int "
            _opts+="--num_speakers $(cat "${data_feats}/org/${train_set}/spk2sid" | wc -l)"
        fi

        # Add language ID to the inputs if needed
        if "${use_lid}"; then
            _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2lid,lids,text_int "
            _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2lid,lids,text_int "
        fi

        if [ "${feats_normalize}" = "global_mvn" ]; then
            _opts+="--normalize_conf stats_file=${tts_stats_dir}/train/feats_stats.npz "
        fi

        log "Generate '${tts_exp}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${tts_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${tts_exp}/run.sh"; chmod +x "${tts_exp}/run.sh"

        log "TTS training started... log: '${tts_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${tts_exp})"
        else
            jobname="${tts_exp}/train.log"
        fi

        echo "$_opts" > /tmp/opts
        
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${tts_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${tts_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m "espnet2.bin.${tts_task}_train" \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --normalize "${feats_normalize}" \
                --resume true \
                --output_dir "${tts_exp}" \
                --allow_variable_data_keys true \
                --keep_nbest_models 10 \
                ${_opts} ${train_args}

    fi
else
    log "Skip training stages"
fi


if [ -n "${download_model}" ]; then
    log "Use ${download_model} for decoding and evaluation"
    tts_exp="${expdir}/${download_model}"
    mkdir -p "${tts_exp}"

    # If the model already exists, you can skip downloading
    espnet_model_zoo_download --unpack true "${download_model}" > "${tts_exp}/config.txt"

    # Get the path of each file
    _model_file=$(<"${tts_exp}/config.txt" sed -e "s/.*'model_file': '\([^']*\)'.*$/\1/")
    _train_config=$(<"${tts_exp}/config.txt" sed -e "s/.*'train_config': '\([^']*\)'.*$/\1/")

    # Create symbolic links
    ln -sf "${_model_file}" "${tts_exp}"
    ln -sf "${_train_config}" "${tts_exp}"
    inference_model=$(basename "${_model_file}")

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        log "Stage 10: Decoding: training_dir=${tts_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi

        _scp=$feats_file
        # if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        # else
        #     # "sound" supports "wav", "flac", etc.
        #     _type=sound
        # fi

        log "Generate '${tts_exp}/${inference_tag}/run.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${tts_exp}/${inference_tag}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${tts_exp}/${inference_tag}/run.sh"; chmod +x "${tts_exp}/${inference_tag}/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _speech_data="${_data}"
            _dir="${tts_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/log"
            mkdir -p "${_logdir}"

            _ex_opts=""
            if [ -n "${teacher_dumpdir}" ]; then
                # Use groundtruth of durations
                _teacher_dir="${teacher_dumpdir}/${dset}"
                _opts+="--data_path_and_name_and_type ${data_feats}/${dset}/durations.scp,durations,kaldi_ark "
#                _opts+="--data_path_and_name_and_type ${_data}/pitch_clusters.scp,pitch,kaldi_ark "
#                _opts+="--data_path_and_name_and_type ${_data}/energy_clusters.scp,energy,kaldi_ark "
            fi
            
#            _opts+="--data_path_and_name_and_type ${data_feats}/${dset}/phone_word_mappings,phone_word_mappings,text_int   " 
#            _opts+="--data_path_and_name_and_type ${data_feats}/${dset}/feats_word_avg.scp,feats_word_avg,kaldi_ark  " 

            # Add X-vector to the inputs if needed
            if "${use_xvector}"; then
                _xvector_dir="${data_feats}/${dset}"
                _ex_opts+="--data_path_and_name_and_type ${_xvector_dir}/xvector.scp,spembs,kaldi_ark "
            fi

            # Add spekaer ID to the inputs if needed
            if "${use_sid}"; then
                _ex_opts+="--data_path_and_name_and_type ${_data}/utt2sid,sids,text_int "
                _opts+="--num_speakers $(cat "${data_feats}/org/${train_set}/spk2sid" | wc -l)"
            fi

            # Add language ID to the inputs if needed
            if "${use_lid}"; then
                _ex_opts+="--data_path_and_name_and_type ${_data}/utt2lid,lids,text_int "
            fi

            # 0. Copy feats_type
            cp "${_data}/feats_type" "${_dir}/feats_type"

            # 1. Split the key file
            key_file=${_data}/text
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 3. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/tts_inference.*.log'  ${_speech_data}/${_scp},speech,${_type} "
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:${_nj} "${_logdir}"/tts_inference.JOB.log \
                ${python} -m espnet2.bin.tts_inference \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/text,text,text" \
                    --data_path_and_name_and_type ${_speech_data}/${_scp},speech,${_type} \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --model_file "${tts_exp}"/"${inference_model}" \
                    --train_config "${tts_exp}"/config.yaml \
                    --output_dir "${_logdir}"/output.JOB \
                    --vocoder_file "${vocoder_file}" \
                    ${_opts} ${_ex_opts} ${inference_args}

            # 4. Concatenates the output files from each jobs
            if [ -e "${_logdir}/output.${_nj}/norm" ]; then
                mkdir -p "${_dir}"/norm
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/norm/feats.scp"
                done | LC_ALL=C sort -k1 > "${_dir}/norm/feats.scp"
            fi
            if [ -e "${_logdir}/output.${_nj}/denorm" ]; then
                mkdir -p "${_dir}"/denorm
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/denorm/feats.scp"
                done | LC_ALL=C sort -k1 > "${_dir}/denorm/feats.scp"
            fi
            if [ -e "${_logdir}/output.${_nj}/speech_shape" ]; then
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/speech_shape/speech_shape"
                done | LC_ALL=C sort -k1 > "${_dir}/speech_shape"
            fi
            if [ -e "${_logdir}/output.${_nj}/wav" ]; then
                mkdir -p "${_dir}"/wav
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
                    rm -rf "${_logdir}/output.${i}"/wav
                done
                find "${_dir}/wav" -name "*.wav" | while read -r line; do
                    echo "$(basename "${line}" .wav) ${line}"
                done | LC_ALL=C sort -k1 > "${_dir}/wav/wav.scp"
            fi
            if [ -e "${_logdir}/output.${_nj}/att_ws" ]; then
                mkdir -p "${_dir}"/att_ws
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/att_ws/*.png "${_dir}"/att_ws
                    rm -rf "${_logdir}/output.${i}"/att_ws
                done
            fi
            if [ -e "${_logdir}/output.${_nj}/durations" ]; then
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/durations/durations"
                done | LC_ALL=C sort -k1 > "${_dir}/durations"
            fi
            if [ -e "${_logdir}/output.${_nj}/focus_rates" ]; then
                for i in $(seq "${_nj}"); do
                     cat "${_logdir}/output.${i}/focus_rates/focus_rates"
                done | LC_ALL=C sort -k1 > "${_dir}/focus_rates"
            fi
            if [ -e "${_logdir}/output.${_nj}/probs" ]; then
                mkdir -p "${_dir}"/probs
                for i in $(seq "${_nj}"); do
                    mv -u "${_logdir}/output.${i}"/probs/*.png "${_dir}"/probs
                    rm -rf "${_logdir}/output.${i}"/probs
                done
            fi
        done
    fi
else
    log "Skip the evaluation stages"
fi

packed_model="${tts_exp}/${tts_exp##*/}_${inference_model%.*}.zip"
if [ -z "${download_model}" ]; then
    # Skip pack preparation if using a downloaded model
    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        log "Stage 11: Pack model: ${packed_model}"
        log "Warning: Upload model to Zenodo will be deprecated. We encourage to use Hugging Face"

        _opts=""
        if [ -e "${tts_stats_dir}/train/feats_stats.npz" ]; then
            _opts+=" --option ${tts_stats_dir}/train/feats_stats.npz"
        fi
        if [ -e "${tts_stats_dir}/train/pitch_stats.npz" ]; then
            _opts+=" --option ${tts_stats_dir}/train/pitch_stats.npz"
        fi
        if [ -e "${tts_stats_dir}/train/energy_stats.npz" ]; then
            _opts+=" --option ${tts_stats_dir}/train/energy_stats.npz"
        fi
        if "${use_xvector}"; then
            for dset in "${train_set}" ${test_sets}; do
                _opts+=" --option ${dumpdir}/xvector/${dset}/spk_xvector.scp"
                _opts+=" --option ${dumpdir}/xvector/${dset}/spk_xvector.ark"
            done
        fi
        if "${use_sid}"; then
            _opts+=" --option ${data_feats}/org/${train_set}/spk2sid"
        fi
        if "${use_lid}"; then
            _opts+=" --option ${data_feats}/org/${train_set}/lang2lid"
        fi
        ${python} -m espnet2.bin.pack tts \
            --train_config "${tts_exp}"/config.yaml \
            --model_file "${tts_exp}"/"${inference_model}" \
            --option "${tts_exp}"/images  \
            --outpath "${packed_model}" \
            ${_opts}

        # NOTE(kamo): If you'll use packed model to inference in this script, do as follows
        #   % unzip ${packed_model}
        #   % ./run.sh --stage 8 --tts_exp $(basename ${packed_model} .zip) --inference_model pretrain.pth
    fi
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    log "Stage 12: Training student model"

     _train_dir="${data_feats}/${train_set}"
    _valid_dir="${data_feats}/${valid_set}"
    log "Stage 12: TTS Student Training: train_set=${_train_dir}, valid_set=${_valid_dir}, config=${student_train_config}"

    _opts="--config ${student_train_config}  "

    if "${use_xvector}"; then
        _xvector_train_dir="${data_feats}/${train_set}"
        _xvector_valid_dir="${data_feats}/${valid_set}"
        _opts+="--train_data_path_and_name_and_type ${_xvector_train_dir}/xvector.scp,spembs,kaldi_ark "
        _opts+="--valid_data_path_and_name_and_type ${_xvector_valid_dir}/xvector.scp,spembs,kaldi_ark "
    fi

    _teacher_train_dir="${teacher_dumpdir}/${train_set}"
    _teacher_valid_dir="${teacher_dumpdir}/${valid_set}"
    
    _opts+="--train_data_path_and_name_and_type ${_train_dir}/text,text,text "
    _opts+="--train_shape_file ${tts_stats_dir}/train/text_shape.${token_type} "
    _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/text,text,text "
    _opts+="--valid_shape_file ${tts_stats_dir}/valid/text_shape.${token_type} "

    _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/feats.scp,speech,kaldi_ark "
    _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/feats.scp,speech,kaldi_ark "

    _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/durations,durations,text_int "
    _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/durations,durations,text_int "
    _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/pitch,pitch,npy "
    _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/pitch,pitch,npy "
    _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/energy,energy,npy "
    _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/energy,energy,npy "
#    _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/feats_word_avg.scp,feats_word_avg,kaldi_ark  " \
#    _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/feats_word_avg.scp,feats_word_avg,kaldi_ark  " \
#    _opts+="--train_data_path_and_name_and_type ${data_feats}/${train_set}/phone_word_mappings,phone_word_mappings,text_int   " \
#    _opts+="--valid_data_path_and_name_and_type ${data_feats}/${valid_set}/phone_word_mappings,phone_word_mappings,text_int   " \
    _opts+="--odim ${odim} "

    if [ -e ${_teacher_train_dir}/probs ]; then
        # Knowledge distillation case: use the outputs of the teacher model as the target
        _scp=$feats_file
        _type=npy
    else
        # Teacher forcing case: use groundtruth as the target
        _scp=wav.scp
        _type=sound
    fi


    # Add speaker ID to the inputs if needed
    if "${use_sid}"; then
        _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2sid,sids,text_int "
        _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2sid,sids,text_int "
        _opts+="--num_speakers $(cat "${data_feats}/org/${train_set}/spk2sid" | wc -l)"
    fi

    # Add language ID to the inputs if needed
    if "${use_lid}"; then
        _opts+="--train_data_path_and_name_and_type ${_train_dir}/utt2lid,lids,text_int "
        _opts+="--valid_data_path_and_name_and_type ${_valid_dir}/utt2lid,lids,text_int "
    fi

    if [ "${feats_normalize}" = "global_mvn" ]; then
        _opts+="--normalize_conf stats_file=${tts_stats_dir}/train/feats_stats.npz "
    fi

    log "Generate '${tts_exp}/run.sh'. You can resume the process from stage 6 using this script"
    mkdir -p "${tts_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${tts_exp}/run.sh"; chmod +x "${tts_exp}/run.sh"

    log "TTS training started... log: '${tts_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${tts_exp})"
    else
        jobname="${tts_exp}/train.log"
    fi

    echo "$_opts" > /tmp/opts
    
    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${student_tts_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${student_tts_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        ${python} -m "espnet2.bin.${tts_task}_train" \
            --use_preprocessor true \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --normalize "${feats_normalize}" \
            --resume true \
            --output_dir "${student_tts_exp}" \
            --allow_variable_data_keys true \
            ${_opts} ${train_args}

fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    log "Stage 13: Decoding: training_dir=${student_tts_exp}"

    if ${gpu_inference}; then
        _cmd="${cuda_cmd}"
        _ngpu=1
    else
        _cmd="${decode_cmd}"
        _ngpu=0
    fi

    _opts=
    if [ -n "${inference_config}" ]; then
        _opts+="--config ${inference_config} "
    fi

    _scp=$feats_file
    # if [[ "${audio_format}" == *ark* ]]; then
        _type=kaldi_ark
    # else
    #     # "sound" supports "wav", "flac", etc.
    #     _type=sound
    # fi


    for dset in ${test_sets}; do
        _data="${data_feats}/${dset}"
        _speech_data="${_data}"
        _dir="${student_tts_exp}/${inference_tag}/${dset}"
        _logdir="${_dir}/log"
        mkdir -p "${_logdir}"

        _ex_opts=""
        if [ -n "${teacher_dumpdir}" ]; then
            # Use groundtruth of durations
            _teacher_dir="${teacher_dumpdir}/${dset}"
            _opts+="--data_path_and_name_and_type ${data_feats}/${dset}/durations,durations,text_int "
            _opts+="--data_path_and_name_and_type ${_data}/pitch,pitch,npy "
            _opts+="--data_path_and_name_and_type ${_data}/energy,energy,npy "
        fi
        
#        _opts+="--data_path_and_name_and_type ${data_feats}/${dset}/phone_word_mappings,phone_word_mappings,text_int   " \
#        _opts+="--data_path_and_name_and_type ${data_feats}/${dset}/feats_word_avg.scp,feats_word_avg,kaldi_ark  " \

        # Add X-vector to the inputs if needed
        if "${use_xvector}"; then
            _xvector_dir="${data_feats}/${dset}"
            _ex_opts+="--data_path_and_name_and_type ${_xvector_dir}/xvector.scp,spembs,kaldi_ark "
        fi

        # Add spekaer ID to the inputs if needed
        if "${use_sid}"; then
            _ex_opts+="--data_path_and_name_and_type ${_data}/utt2sid,sids,text_int "
            _opts+="--num_speakers $(cat "${data_feats}/org/${train_set}/spk2sid" | wc -l)"
        fi

        # Add language ID to the inputs if needed
        if "${use_lid}"; then
            _ex_opts+="--data_path_and_name_and_type ${_data}/utt2lid,lids,text_int "
        fi

        # 0. Copy feats_type
        cp "${_data}/feats_type" "${_dir}/feats_type"

        # 1. Split the key file
        key_file=${_data}/text
        split_scps=""
        _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/keys.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 3. Submit decoding jobs
        log "Decoding started... log: '${_logdir}/tts_inference.*.log'  ${_speech_data}/${_scp},speech,${_type} "
        # shellcheck disable=SC2086
        ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/tts_inference.JOB.log \
            ${python} -m espnet2.bin.tts_inference \
                --ngpu "${_ngpu}" \
                --data_path_and_name_and_type "${_data}/text,text,text" \
                --data_path_and_name_and_type ${_speech_data}/${_scp},speech,${_type} \
                --key_file "${_logdir}"/keys.JOB.scp \
                --model_file "${student_tts_exp}"/"${inference_model}" \
                --train_config "${student_tts_exp}"/config.yaml \
                --output_dir "${_logdir}"/output.JOB \
                --vocoder_file "${vocoder_file}" \
                ${_opts} ${_ex_opts} ${inference_args}

        # 4. Concatenates the output files from each jobs
        if [ -e "${_logdir}/output.${_nj}/norm" ]; then
            mkdir -p "${_dir}"/norm
            for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/norm/feats.scp"
            done | LC_ALL=C sort -k1 > "${_dir}/norm/feats.scp"
        fi
        if [ -e "${_logdir}/output.${_nj}/denorm" ]; then
            mkdir -p "${_dir}"/denorm
            for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/denorm/feats.scp"
            done | LC_ALL=C sort -k1 > "${_dir}/denorm/feats.scp"
        fi
        if [ -e "${_logdir}/output.${_nj}/speech_shape" ]; then
            for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/speech_shape/speech_shape"
            done | LC_ALL=C sort -k1 > "${_dir}/speech_shape"
        fi
        if [ -e "${_logdir}/output.${_nj}/wav" ]; then
            mkdir -p "${_dir}"/wav
            for i in $(seq "${_nj}"); do
                mv -u "${_logdir}/output.${i}"/wav/*.wav "${_dir}"/wav
                rm -rf "${_logdir}/output.${i}"/wav
            done
            find "${_dir}/wav" -name "*.wav" | while read -r line; do
                echo "$(basename "${line}" .wav) ${line}"
            done | LC_ALL=C sort -k1 > "${_dir}/wav/wav.scp"
        fi
        if [ -e "${_logdir}/output.${_nj}/att_ws" ]; then
            mkdir -p "${_dir}"/att_ws
            for i in $(seq "${_nj}"); do
                mv -u "${_logdir}/output.${i}"/att_ws/*.png "${_dir}"/att_ws
                rm -rf "${_logdir}/output.${i}"/att_ws
            done
        fi
        if [ -e "${_logdir}/output.${_nj}/durations" ]; then
            for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/durations/durations"
            done | LC_ALL=C sort -k1 > "${_dir}/durations"
        fi
        if [ -e "${_logdir}/output.${_nj}/focus_rates" ]; then
            for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/focus_rates/focus_rates"
            done | LC_ALL=C sort -k1 > "${_dir}/focus_rates"
        fi
        if [ -e "${_logdir}/output.${_nj}/probs" ]; then
            mkdir -p "${_dir}"/probs
            for i in $(seq "${_nj}"); do
                mv -u "${_logdir}/output.${i}"/probs/*.png "${_dir}"/probs
                rm -rf "${_logdir}/output.${i}"/probs
            done
        fi
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
