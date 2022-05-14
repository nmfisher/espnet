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
        "durations", type=str, help="Path to durations SCP (e.g. 'durations')"
    )
    parser.add_argument(
        "text", type=str, help="Path to text SCP (e.g. 'text')"
    )
    parser.add_argument(
        "feats_out", type=str, help="wspecifier for outputs (e.g. 'ark,scp:feats.ark,feats.scp')"
    )
    parser.add_argument(
        "phone_word_mappings_out", type=str, help="wspecifier for writing the word index of every phone (e.g. 'ark,scp:phone_word_indices.ark,phone_word_indices.scp')"
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

    durations = _open(args.durations)
    transcripts = _open(args.text)
    # all_feats = _open(args.feats)
    
    with ReadHelper("scp:" + args.feats) as feats_reader:
            with WriteHelper(args.feats_out) as writer:
                with WriteHelper(args.phone_word_mappings_out) as phone_word_mapping_writer:
                    print(f"Using {args.word_phone_mappings_out}")
                    with WriteHelper(args.word_phone_mappings_out,compression_method=2) as word_phone_mapping_writer:
                        for ((utt_id, durations), (_,phones,), (_, feats), (_,pitch_file)) in zip(durations, transcripts, feats_reader, _open(args.pitch)):
                            
                            # this is a list of W elements (where W is the number of words) where each element is the index in [phones] that marks the end of a word
                            # accordingly, [phones] is a phonetic transcription of N words
                            P = len(phones)
                            
                            # pitch = np.fromfile(pitch_file[0])
                            phone_indices = np.arange(P)
                            word_boundaries = phone_indices[[x for x in range(len(phones)) if phones[x].endswith("E") or phones[x] in ["SPN","SIL"]]]
                            W = len(word_boundaries)

                            # use this to create a list of P elements (where P is the length of [phones]), 
                            # where each element is the word index that the phone at that index belongs to
                            phone_word_mappings = []

                            # also use this to create a one-hot encoded list of WxP elements, i.e. where each row is a word, and each column is a phone 
                            word_phone_mappings = np.zeros((W, P),dtype=np.int32)

                            # also use this to average the features according to word_boundaries
                            feats_avg = []                    
                            
                            word_idx = 0
                            durations = [int(x) for x in durations]
                            #print(f"Duratons : {durations}")
                            #print(f"feats {feats.shape}")
                            #print(f"word_boundaries {word_boundaries}")
                            for i in range(len(word_boundaries)):
                                word_end_idx = word_boundaries[i]
                                word_start_idx = word_boundaries[i-1]+1 if i > 0 else 0
                                num_phones = 1 + word_end_idx - word_start_idx
                                num_frames = sum(durations[word_start_idx:word_end_idx+1])
                                frame_offset = sum(durations[:word_start_idx]) if i > 0 else 0
                                # print(f"num_frames {num_frames} frame_offset {frame_offset}")    
                                word_feats = np.array(feats[frame_offset:frame_offset+num_frames])

                                stacked = word_feats #np.hstack([word_feats,word_pitch])
                                                                
                                feats_avg += [ stacked.mean(0) ]
                                if np.any(np.isnan(feats_avg)):
                                    raise Exception()

                                phone_word_mappings += [word_idx] * num_phones
                                word_phone_mappings[word_idx,word_start_idx:word_end_idx+1] = 1
                                
                                word_idx += 1
                                
                            if len(phone_word_mappings) != len(phones):
                                raise Exception(f"Word mapping mismatch for {utt_id} phones length {len(phones)} {phones}, durations {durations} word boundaries were {word_boundaries}, mappings {len(phone_word_mappings)} were {phone_word_mappings}")
                            
                            writer(utt_id, np.vstack(feats_avg))
                            phone_word_mapping_writer(utt_id, np.array(phone_word_mappings,dtype=np.int32))
                            word_phone_mapping_writer(utt_id, word_phone_mappings)


if __name__ == "__main__":
    main()
