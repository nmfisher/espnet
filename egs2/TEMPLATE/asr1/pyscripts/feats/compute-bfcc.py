#!/usr/bin/env python3.7
import argparse
import logging
import sys
import numpy as np
from espnet.utils.cli_utils import get_commandline_args

import scipy.io.wavfile
from spafe.features.bfcc import bfcc
from kaldiio import ReadHelper, WriteHelper
import subprocess
import tempfile

def get_parser():
    parser = argparse.ArgumentParser(
        description="computes Bark-scale cepstral coefficients from raw waveform. Important - input audio should be sampled at 16kHz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "wavs", type=str, help="rspecifier for inputs (e.g. 'scp:wav.scp')"
    )
    parser.add_argument(
        "outs", type=str, help="wspecifier for outputs (e.g. 'ark,scp:feats.ark,feats.scp')"
    )
    parser.add_argument(
        "sample_rate", type=int, help="target sample rate"
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

    with WriteHelper(args.outs) as writer:
      with ReadHelper(args.wavs) as reader:
            for utt_id, (fs, audio) in reader: 
                if fs != args.sample_rate:
                    raise Exception("Expected sample rate %d but %d was provided, please make sure all audio inputs are resampled first." % (args.sample_rate, fs))
 
                with tempfile.NamedTemporaryFile() as pcm_out:
                    with tempfile.NamedTemporaryFile() as feats_out:
                        pcm_out.write(audio)
                        process = subprocess.run(['lpcnet_demo', '-features', pcm_out.name, feats_out.name])
                        feats = np.fromfile(feats_out.name, dtype=np.float32).reshape(-1,36)
                        writer(utt_id, feats[:,:20])
                        print(f"Wrote to feats file {feats_out.name}")


if __name__ == "__main__":
    main()
