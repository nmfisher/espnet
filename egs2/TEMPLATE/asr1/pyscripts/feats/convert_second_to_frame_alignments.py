#!/usr/bin/env python3.7
import argparse
import logging
import sys
import math 
import numpy as np
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import get_commandline_args
from kaldiio import ReadHelper, WriteHelper

def get_parser():
    parser = argparse.ArgumentParser(
        description="Converts durations from seconds to frames. Accounts for rounding to allocate appropriately.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "durations", type=str, help="rspecifier for input durations (in seconds). e.g. ark,t:data/train/durations"
    )
    parser.add_argument(
        "feats", type=str, help="rspecifier for feats (used to check the number of frames matches exactly). e.g. scp:data/train/feats.scp"
    )
    parser.add_argument(
        "output", type=str, help="wspecifier for output durations (in frames). e.g. ark,t:data/train/durations"
    )
    parser.add_argument(
        "framerate", type=int, help="Number of frames per second"
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f"feats {args.feats} durs {args.durations} output {args.output}")
    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.info(get_commandline_args())
    with ReadHelper(args.feats) as feats_reader, ReadHelper(args.durations) as durations_reader, WriteHelper(args.output) as writer:
        for (utt_id, durs), (feat_utt_id, feats) in zip(durations_reader, feats_reader):
            assert utt_id == feat_utt_id
            print(feats.shape)
            print(sum(durs))            
            print(durs)

            durs = durs * args.framerate
            frame_durations=[]
            accum = 0
            
            for dur in durs:
                dur += accum
                accum = dur - math.floor(dur)
                frame_durations += [ math.floor(dur) ]
            sum_frames=sum(frame_durations)
            
            feat_frames = feats.shape[0]
            diff = sum_frames - feat_frames
            
            if diff > 5 or diff < -5:
                raise Exception(f"Frame diff {diff} too big for {utt_id}")
            elif diff > 0:
                for i in range(len(frame_durations) - 1, 0, -1):
                    if frame_durations[i] > 1:
                        frame_durations[i] -= 1
                        diff -=1
                    if diff == 0:
                        break
            elif diff < 0:
                for i in range(len(frame_durations) - 1, 0, -1):
                    frame_durations[i] += 1
                    diff +=1
                    if diff == 0:
                        break    
            
            assert sum(frame_durations) == feats.shape[0], f"{sum(frame_durations)} vs {feat_frames} for utt {utt_id}"
            writer(utt_id, np.array(frame_durations))

if __name__ == "__main__":
    main()
