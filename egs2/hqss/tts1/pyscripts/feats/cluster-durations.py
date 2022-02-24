#!/usr/bin/env python3.7
import argparse
import logging
import sys
import numpy as np
from espnet.transform.transformation import Transformation
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_utils import is_scipy_wav_style


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

def write_clusters(filepath, durations, transcripts, clusters, num_clusters):
  written = 0
  with open(filepath, "w") as outfile:
    for (utt_id, utt_durations), (_, utt_phones) in zip(durations, transcripts):
      cluster_ids = []
      for phone, duration in zip(utt_phones, utt_durations):
        for cluster_id in range(num_clusters):

          if (cluster_id == 0 and duration <= clusters[phone][cluster_id]) or \
              (cluster_id == num_clusters - 1 and duration >= clusters[phone][cluster_id]) or \
              (duration <= clusters[phone][cluster_id] and duration > clusters[phone][cluster_id-1]):
              cluster_ids.append(str(cluster_id))
              break
          else:
            continue
      outfile.write("{} {}\n".format(utt_id, " ".join(cluster_ids)))
      written += 1
  print("Wrote clusters for %d utterances to %s" % (written, filepath))

def _open(path):
  with open(path, "r") as infile:
    try:
        ln=0
        for line in infile.readlines():
          split = line.strip().split(" ")
          yield (split[0], split[1:])
          ln += 1
    except:
        raise Exception(f"Error parsing line {ln} in file {path}")

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
    
    print("Opening train transcripts at %s", args.train_transcripts)
    train_transcripts = list(_open(args.train_transcripts))
    valid_transcripts = list(_open(args.valid_transcripts))
    phone_durations = {}
    count = 0
    for (utt_id, utt_durations), (_, transcript) in zip(train_durations, train_transcripts):
        if len(transcript) != len(utt_durations):
            print(utt_durations)
            raise Exception("%s : %d %d" % (utt_id, len(transcript), len(utt_durations)))
        for phone, duration in zip(transcript, utt_durations):
            if phone not in phone_durations:
                phone_durations[phone] = []
            phone_durations[phone].append(duration)
        count+=1
    print("Processed %d utterances" % count)
    
    # sort the list of durations for each phone and bucket into [num_clusters] s.t. each bucket contains an equal number of samples
    # produces a list of floats [num_clusters] long
    # each float corresponds to the high end of the cluster duration (and each index corresponds to the cluster label)
    # e.g. {
    #  "h":[10,15,27],
    #  "e":[20,45,57], (so cluster 0 <= 20, cluster 1 <= 45, etc)
    # etc
    # }
    # print(phone_durations)
    phone_clusters = {}
    for phone, durations in phone_durations.items():
      durations.sort()
      if len(durations) >= args.num_clusters:
        clusters = durations[::len(durations) // args.num_clusters]
        phone_clusters[phone] = clusters
      else:
        phone_clusters[phone] = list(range(args.num_clusters))

    # finally, iterate over the original dataset against to replace the phone durations with the phone cluster IDs
    # results are written directly to the output file
    write_clusters(args.train_outfile, train_durations, train_transcripts, phone_clusters, args.num_clusters)
    write_clusters(args.valid_outfile, valid_durations, valid_transcripts, phone_clusters, args.num_clusters)
    
if __name__ == "__main__":
    main()
