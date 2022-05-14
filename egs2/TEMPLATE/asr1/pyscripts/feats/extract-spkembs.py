#!/usr/bin/env python3.7
from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020

import argparse
import logging
import sys
import numpy as np
from espnet.utils.cli_utils import get_commandline_args

import scipy.io.wavfile
from kaldiio import ReadHelper, WriteHelper
import subprocess

def get_parser():
    parser = argparse.ArgumentParser(
        description="computes (averaged) speaker embeddings for dataset using pretrained BYOL-A.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "wavs", type=str, help="Path to wav.scp (no scp prefix needed)"
    )
    parser.add_argument(
        "outs", type=str, help="wspecifier for outputs (e.g. 'ark,scp:xvectors.ark,xvector.scp')"
    )
    parser.add_argument(
        "utt2spk", type=str, help="utt<->spk lookup"
    )
    parser.add_argument(
        "--stats", type=str, help="path to existing mel spec mean/stdev. Must be provided if --stats_out is not provided", required=False
    )
    parser.add_argument(
        "--stats_out", type=str, help="path where mel spec mean/stdev. Must be provided if --stats_out is not provided", required=False
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    device = torch.device('cuda')
    cfg = load_yaml_config('/home/hydroxide/projects/byol-a/config.yaml')

    utt2spk = {}
    for line in open(args.utt2spk, "r").readlines():
        split = line.strip().split(" ")
        utt2spk[split[0]] = split[1]

    # Preprocessor and normalizer.
    to_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )

    spkr_vectors={}
    spkr_counts={}

    vals=[]

    if args.stats is None:
        print(f"Calculating stats, will be written to {args.stats_out}")
        if args.stats_out is None:
            raise Exception("--stats_out must be specified if --stats is empty")
        # first pass through to calculate mean and stddev
        count=0
        for line in open(args.wavs).readlines():
            split=line.strip().split(" ")
            utt_id = split[0]
            
            wav, sr = torchaudio.load(split[1]) # a sample from SPCV2 for now
            if sr != cfg.sample_rate:
                raise Exception("Expected sample rate %d but %d was provided for %s, please make sure all audio inputs are resampled first." % (cfg.sample_rate, sr, utt_id))
            melspec = to_melspec(wav).reshape(-1)
            
            vals += [ (melspec + torch.finfo(torch.float).eps).log() ]
        all = np.concatenate(vals)
        stats = np.array([all.mean(), all.std()])
        stats.tofile(args.stats_out)
        print(f"Wrote stats to {args.stats_out}")
        stats = np.fromfile(args.stats_out,dtype=np.float32)
    else:
        stats = np.fromfile(args.stats,dtype=np.float32)
    
    # Mean and standard deviation of the log-mel spectrogram of input audio samples, pre-computed.
    # See calc_norm_stats in evaluate.py for your reference.
    # Load pretrained weights.
    model = AudioNTT2020(d=cfg.feature_d)
    model.load_weight('/home/hydroxide/projects/byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth', device)
    model.eval()
    
    normalizer = PrecomputedNorm(stats)

    utt_ids = []

    with WriteHelper(args.outs) as writer:
        for line in open(args.wavs).readlines():
            split=line.strip().split(" ")
            utt_id = split[0]

            utt_ids.append(utt_id)
            
            wav, sr = torchaudio.load(split[1]) # a sample from SPCV2 for now
            if sr != cfg.sample_rate:
                raise Exception("Expected sample rate %d but %d was provided for %s, please make sure all audio inputs are resampled first." % (cfg.sample_rate, sr, utt_id))
            melspec = to_melspec(wav)
        #     # Convert to a log-mel spectrogram, then normalize.
            lms = normalizer((melspec + torch.finfo(torch.float).eps).log())

            # Now, convert the audio to the representation.
            spkr = utt2spk[utt_id]
            vec = model(lms.unsqueeze(0)).cpu().detach().numpy()

            if spkr not in spkr_vectors:
                spkr_vectors[spkr] = {
                    "count":1,
                    "vector":vec
                }
            else:
                spkr_vectors[spkr]["vector"] += vec
                spkr_vectors[spkr]["count"] += 1
        
        for utt_id in utt_ids:
            averaged = spkr_vectors[utt2spk[utt_id]]["vector"] / spkr_vectors[utt2spk[utt_id]]["count"]
            print(averaged)
            writer(utt_id, averaged)
    
if __name__ == "__main__":
    main()
