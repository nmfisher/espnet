#!/usr/bin/env python3.7
from functools import reduce
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


def get_parser():
    parser = argparse.ArgumentParser(
        description="converts frame-level F0 values to K-means centroids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "sample_rate",
        type=int,
        default=16000,
        help="hop_length used during feat extraction (this is needed as this script accepts ms durations as input, which are then converted to frames"
    )
    parser.add_argument(
        "hop_length",
        type=int,
        default=256,
        help="hop_length used during feat extraction (this is needed as this script accepts ms durations as input, which are then converted to frames"
    )
    parser.add_argument(
        "num_clusters", type=int, help="Number of clusters to use for durations"
    )
    parser.add_argument(
        "train_wav", type=str, help="Path to train WAV file. e.g. data/train/wav.scp"
    )
    parser.add_argument(
        "train_durations", type=str, help="Path to train durations file. e.g. data/train/durations"
    )
    parser.add_argument(
        "train_transcripts", type=str, help="Path to train transcripts. e.g. data/train/text"
    )
    parser.add_argument(
        "train_outfile", type=str, help="Output path for train duration cluster IDs data/train/cluster_ids"
    )
    parser.add_argument(
        "valid_wav", type=str, help="Path to valid WAV file. e.g. data/valid/wav.scp"
    )
    parser.add_argument(
        "valid_durations", type=str, help="Path to validation durations file. e.g. data/valid/durations"
    )
    parser.add_argument(
        "valid_transcripts", type=str, help="Path to valid transcripts. e.g. data/valid/text"
    )
    parser.add_argument(
        "valid_outfile", type=str, help="Output path for validation duration cluster IDs data/train/cluster_ids"
    )
    parser.add_argument(
        "f0min",
        type=int,
        default=70,
        help="Min frequency",
    )
    parser.add_argument(
        "f0max",
        type=int,
        default=8000,
        help="Max frequency",
    )
    return parser


def _open(path):
    with open(path, "r") as infile:
        for line in infile.readlines():
            split = line.strip().split(" ")
            yield (split[0], split[1:])


def extract(wavs, durations, transcripts, sample_rate, hop_length,f0_values):

    for (utt_id, path), (_, utt_durations), (_, transcript) in zip(wavs, durations, transcripts):

        wav, rate = librosa.load(path[0], sr=sample_rate)
        f0, t = pyworld.dio(wav.astype(np.double), rate,
                            frame_period=sample_rate / hop_length)

        utt_f0 = []

        # change the phone durations into start/end timestamps such that
        # (start_time,end_time) for phone[i] is (phone_times[i-1],phone_times[i])
        phone_times = reduce(lambda accum, x: accum
                             + [x + accum[-1]], [float(x) for x in utt_durations], [0])

        start = 0
        # iterate over every phone timestamp
        for time, phone in zip(phone_times[1:], transcript):
            # iterate from the last F0 timestamp until we find a time that's higher (i.e. later) than the phone timestamp
            for i in range(start, f0.shape[0], 1):
                if time <= t[i] or i == f0.shape[0] - 1:
                    # add the F0 value (or simple average, if it spanned more than 1 frame)
                    phone_f0 = (f0[i] + f0[start]) / \
                        (i - start) if i != start else f0[i]
                    utt_f0.append(phone_f0)
                    start = i
                    break

        if len(utt_f0) != len(transcript):
            #print(f"{len(phone_f0)} {len(phone_times)}")
            print(f"{path}\n transcript : {transcript}\n f0 : {utt_f0}\n DIO timestamps {t}\n phone_times {phone_times}\n original durations: {utt_durations}")
            raise Exception("length mismatch")
            # if phone_times[-1] > t[-1]:
        #    phone_f0.append((f0[-1] + f0[start]) / (len(f0) - start))
        yield (utt_id, utt_f0)


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

    # for each audio file in the training dataset
    # load & extract F0 for each phone (using provided durations)
    f0 = []
    train_f0 = list(extract(_open(args.train_wav), _open(
        args.train_durations), _open(args.train_transcripts), args.sample_rate, args.hop_length, f0))

    valid_f0_values = {}
    valid_f0 = extract(_open(args.valid_wav), _open(
        args.valid_durations), _open(args.valid_transcripts), args.sample_rate, args.hop_length, valid_f0_values)

    # normalize train/valid pitches based on mean/stddev from train only
    # todo - multi-speaker basis
    f0=np.array(list(itertools.chain.from_iterable([v for k,v in train_f0])))
    
    mean =f0.mean(axis = 0)
    stddev =f0.std(axis = 0)
    # obs for clustering
    obs = [(p - mean) / stddev for p in f0]

    # now we cluster
    k=KMeans(args.num_clusters)
    
    k.fit(np.array(obs).reshape(len(obs), 1))

    # normalize training/validation set based on training means/stddevs
    
    train_f0 = [ (utt_id, (f0 - mean) / stddev)  for (utt_id, f0) in train_f0]
    
    valid_f0 = [ (utt_id, (f0 - mean) / stddev)  for (utt_id, f0) in valid_f0]
    
    # finally, iterate over the original dataset again  to replace the F0 values with the phone cluster IDs
    for vals, out in [(train_f0, args.train_outfile), (valid_f0, args.valid_outfile)]:
        print(f"Writing {len(vals)} values to {out}")
        with open(out, "w") as outfile:
            for utt_id, f0 in vals:
                cluster_ids=k.predict(np.array(f0).reshape(len(f0), 1)).tolist()
                cluster_ids=[str(x) for x in cluster_ids]
                outfile.write("{} {}\n".format(utt_id, " ".join(cluster_ids)))

if __name__ == "__main__":
    main()
# obs=[]
#     means={}
#     stddevs={}
#     for phone in [v for k,v in train_f0_values:
#         f0=np.array(train_f0_values[phone])
#         means[phone]=f0.mean(axis = 0)
#         stddevs[phone]=f0.std(axis = 0)

#         # just while we're working with a small dataset
#         if stddevs[phone] == 0:
#             stddevs[phone] += 0.01
#             print(phone)
#         # add observation for clustering purposes
#         obs += [(p - means[phone]) / stddevs[phone] for p in f0]