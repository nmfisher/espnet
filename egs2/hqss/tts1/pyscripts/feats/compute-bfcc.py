#!/usr/bin/env python3
import argparse
import logging
import sys
import numpy as np
from espnet.utils.cli_utils import get_commandline_args

import scipy.io.wavfile
from spafe.features.bfcc import bfcc
from kaldiio import ReadHelper, WriteHelper
import subprocess

def get_parser():
    parser = argparse.ArgumentParser(
        description="converts raw phone durations to clustered labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "wavs", type=str, help="rspecifier for inputs (e.g. 'scp:wav.scp')"
    )
    parser.add_argument(
        "outs", type=str, help="wspecifier for outputs (e.g. 'ark,scp:feats.ark,feats.scp')"
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
              with open("/tmp/tmp.pcm", "wb") as outfile:
                outfile.write(audio)
              process = subprocess.run(['lpcnet_demo', '-features', '/tmp/tmp.pcm', '/tmp/tmp.feats'])
              feats = np.fromfile("/tmp/tmp.feats", dtype=np.float32).reshape(-1,36)

              writer(utt_id, feats[:,:20])


if __name__ == "__main__":
    main()
