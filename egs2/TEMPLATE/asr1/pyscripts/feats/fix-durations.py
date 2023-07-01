#!/usr/bin/env python3.7
from functools import reduce
import torch
import itertools
import argparse
import logging
import sys
import numpy as np
from espnet.transform.transformation import Transformation
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import get_commandline_args
import pyworld
import librosa
from kaldiio import ReadHelper, WriteHelper
from sklearn.cluster import KMeans
from espnet2.layers.stft import Stft

"""
There will occasionally be a slight mismatch between the number of frames in extracted features and the number of frames in the forced alignment.
Usually this is only a single frame, and occurs due to differences in rounding between different toolkits.
This script fixes this by adding/subtracting the frame difference from the first/last phone durations. These are usually SIL/SPN/etc so the difference is not noticeable.
"""

def get_parser():
    parser = argparse.ArgumentParser(
        description="converts frame-level F0 values to K-means centroids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")

    parser.add_argument(
        "train_feats", type=str, help="e.g. scp:data/train/feats.scp"
    )
    parser.add_argument(
        "valid_feats", type=str, help="e.g. scp:data/valid/feats.scp"
    )
    parser.add_argument(
        "train_durations", type=str, help="Path to train durations (frame lengths). e.g. data/train/durations"
    )
    parser.add_argument(
        "train_durations_out", type=str, help="Output path for fixed train durations e.g. dump/raw/train/durations"
    )
    parser.add_argument(
        "valid_durations", type=str, help="Path to valid durations (frame lengths). e.g. data/valid/durations"
    )
    parser.add_argument(
        "valid_durations_out", type=str, help="Output path for fixed validation durations e.g. dump/raw/valid/durations"
   )
    parser.add_argument(
        "odim", type=str, help="Output dimension (needed if feats is 1D but needs to be reshaped to 2D)"
    )
    return parser

def fix(feats_file, durations_file, durations_out_file, odim):
  with open(durations_out_file, "w") as dur_writer:
    print(f"Fixing durations based on feature file {feats_file} and durations file {durations_file}")
    with ReadHelper(feats_file) as feats_reader, ReadHelper(durations_file) as dur_reader:
      for (utt_id, feats), (utt_id2, durations) in zip(feats_reader, dur_reader):
        if utt_id != utt_id2:
          raise Exception(f"Utterance ID mismatch in files {feats_file} and {durations_file} : {utt_id} vs {utt_id2}")
        if len(feats.shape) == 1:
            feats = feats.reshape(-1,odim)
        dsum = durations.sum()
        if dsum < feats.shape[0]:
          durations[-1] += (feats.shape[0] - dsum)
        elif dsum > feats.shape[0]:
          durations[0] -= (dsum - feats.shape[0])
        if int(round(durations.sum())) != feats.shape[0]:
            print(durations)
            raise Exception(f"Mismatch durations : {durations.sum()} vs feats {feats.shape[0]}")
        dur_writer.write("%s %s\n" % (utt_id, " ".join([str(int(x)) for x in durations])))
        print(f"Fixed durations for utterance {utt_id}")

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

    fix(args.train_feats, args.train_durations, args.train_durations_out, int(args.odim))
    fix(args.valid_feats, args.valid_durations, args.valid_durations_out, int(args.odim))


if __name__ == "__main__":
    main()
