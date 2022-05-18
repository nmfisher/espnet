#!/usr/bin/env python3.7
import argparse
import logging
import sys
import numpy as np
from espnet.utils.cli_utils import get_commandline_args

from kaldiio import ReadHelper, WriteHelper

def get_parser():
    parser = argparse.ArgumentParser(
        description="averages frame-level acoustic features by word.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "feats", type=str, help="Path to extracted features in SCP format (e.g. 'feats.scp')"
    )
    parser.add_argument(
        "pitch", type=str, help="Path to extracted pitch features in SCP format (e.g. 'pitch.scp')"
    )
    parser.add_argument(
        "durations", type=str, help="Path to durations (e.g. 'durations') (tab-separated, leftmost column is utterance ID, right column is space-separated list of ints where each int is the duration of the phone at that index"
    )
    parser.add_argument(
        "text", type=str, help="Path to text SCP (e.g. 'text')"
    )
    parser.add_argument(
        "feats_out", type=str, help="wspecifier for outputs (e.g. 'ark,scp:feats.ark,feats.scp')"
    )
    parser.add_argument(
        "phone_word_mappings", type=str, help="Path to file containing phone->word mappings (e.g. 'phone_word_indices') (tab-separated, leftmost column is utterance ID, right column is space-separated list of ints where each int is the index of the word that the phone at that index belongs to"
    )
    parser.add_argument(
        "word_phone_mappings_out", type=str, help="wspecifier for writing the one-hot matrix of phone indices for each word  (e.g. 'ark,scp:word_phone_mappings.ark,word_phone_mappings.scp')"
    )
    return parser


def _open(path):
    print(f"Opening file at path {path}")
    with open(path, "r") as infile:
        i = 0
        for line in infile.readlines():
            i += 1
            split = line.strip().split()
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

    durations = _open(args.durations)
    transcripts = _open(args.text)
    
    with ReadHelper("scp:" + args.feats) as feats_reader:
            with WriteHelper(args.feats_out) as writer:
                print(f"Using {args.word_phone_mappings_out}")
                with WriteHelper(args.word_phone_mappings_out,compression_method=2) as word_phone_mapping_writer:
                        for ((utt_id, durations), \
                            (_,phones,), \
                            (_, feats), \
                            (_,pitch_file), \
                            (_, phone_word_mappings)) in \
                                zip(durations, \
                                    transcripts, \
                                    feats_reader, \
                                    _open(args.pitch), \
                                    _open(args.phone_word_mappings)):
                            
                            phone_word_mappings = np.array([int(x) for x in phone_word_mappings])
                            # this is a list of W elements (where W is the number of words) where each element is the index in [phones] that marks the end of a word
                            # accordingly, [phones] is a phonetic transcription of N words
                            P = len(phones)
                            W = phone_word_mappings.max() + 1

                            # also use this to create a one-hot encoded list of WxP elements, i.e. where each row is a word, and each column is a phone 
                            word_phone_mappings = np.zeros((W, P),dtype=np.int32)

                            # also use this to average the features according to word_boundaries
                            feats_avg = []                    
                            
                            word_idx = 0
                            durations = [int(x) for x in durations]

                            phone_idx = 0

                            # if there's only one word, average across all feat frames
                            if W == 1:
                                feats_avg = [feats.mean(0)]
                            else:
                                for word_idx in range(W):
                                    avg = np.zeros((1, feats.shape[1]))
                                    num_phones = 0
                                    while phone_idx < len(durations) and phone_word_mappings[phone_idx] == word_idx:
                                        frame_offset = sum(durations[:phone_idx])
                                        phone_duration = durations[phone_idx]
                                        avg += np.array(feats[frame_offset:frame_offset+phone_duration]).mean(0)
                                        num_phones += 1
                                        word_phone_mappings[word_idx,phone_idx] = 1
                                        phone_idx += 1
                                    avg /= num_phones
                                    feats_avg += [avg]
                                    word_idx += 1
                                
                            if len(phone_word_mappings) != len(phones):
                                raise Exception(f"Word mapping mismatch for {utt_id} phones length {len(phones)} {phones}, durations {durations} mappings {len(phone_word_mappings)} were {phone_word_mappings}")
                            if len(feats_avg) == 0:
                                raise Exception(f"Empty features for {utt_id}")
                            
                            writer(utt_id, np.vstack(feats_avg))
                            word_phone_mapping_writer(utt_id, word_phone_mappings)


if __name__ == "__main__":
    main()
