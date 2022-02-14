#!/usr/bin/env python3.7
import argparse
import logging
import sys
import numpy as np
from espnet.transform.transformation import Transformation
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_utils import is_scipy_wav_style
from sklearn.cluster import KMeans

def get_parser():
    parser = argparse.ArgumentParser(
        description="converts raw phone durations to clustered labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="hop_length used during feat extraction (this is needed as this script accepts ms durations as input, which are then converted to frames"
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=256,
        help="hop_length used during feat extraction (this is needed as this script accepts ms durations as input, which are then converted to frames"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=5, help="Number of clusters to use for durations"
    )
    parser.add_argument(
        "--train_durations", type=str, help="Path to train durations file. e.g. data/train/durations"
    )
    parser.add_argument(
        "--train_transcripts", type=str, help="Path to train transcripts. e.g. data/train/text"
    )
    parser.add_argument(
        "--train_outfile", type=str, help="Output path for train duration cluster IDs data/train/cluster_ids"
    )
    parser.add_argument(
        "--valid_durations", type=str, help="Path to validation durations file. e.g. data/valid/durations"
    )
    parser.add_argument(
        "--valid_transcripts", type=str, help="Path to valid transcripts. e.g. data/valid/text"
    )
    parser.add_argument(
        "--valid_outfile", type=str, help="Output path for validation duration cluster IDs data/train/cluster_ids"
    )
    parser.add_argument(
        "out",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="The output filename. " "If omitted, then output to sys.stdout",
    )
    return parser

def _open(path):
  with open(path, "r") as infile:
    for line in infile.readlines():
      split = line.strip().split(" ")
      yield (split[0], split[1:])

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

    # first pass through the dataset collects the durations (in frames) for every phone and stores in a map
    # keys are phones, values are list of durations
    # e.g. {
    #  "h":[10,15,...,17,27],
    #  "e":[20,45,...,57,77],
    # ...
    # }

    train_durations = [(x[0], [int((float(d) * args.sample_rate) / args.hop_length) + 1 for d in x[1]]) for x in _open(args.train_durations)]
    valid_durations = [(x[0], [int((float(d) * args.sample_rate) / args.hop_length) + 1 for d in x[1]]) for x in _open(args.valid_durations)]
    
    transcripts = _open(args.train_transcripts)
    durations = []

    for (utt_id, utt_durations), (utt_id, transcript) in zip(train_durations, transcripts):
        if len(transcript) != len(utt_durations):
            print(utt_durations)
            raise Exception("%s : %d %d" % (utt_id, len(transcript), len(utt_durations)))
        for phone, duration in zip(transcript, utt_durations):
            durations.append(duration)
    durations = np.array(durations)
    mean = durations.mean(axis = 0)
    stddev = durations.std(axis = 0)
    # obs for clustering
    obs = [(d - mean) / stddev for d in durations]

    # now we cluster
    k=KMeans(args.num_clusters)
    
    k.fit(np.array(obs).reshape(len(obs), 1))
        
    # finally, iterate over the original dataset against to replace the phone durations with the phone cluster IDs
    # results are written directly to the output
    for out, durations in [(args.train_outfile, train_durations), (args.valid_outfile, valid_durations)]:
      with open(out, "w") as outfile:
        for (utt_id, utt_durations) in durations:
            normalized = np.array([ ((d - mean) / stddev)  for d in utt_durations ])
            cluster_ids=k.predict(normalized.reshape(len(utt_durations), 1)).tolist()
            outfile.write("{} {}\n".format(utt_id, " ".join([str(c) for c in cluster_ids])))

if __name__ == "__main__":
    main()
