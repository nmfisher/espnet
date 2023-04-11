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
        description="converts frame-level F0 & energy values to K-means centroids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--num_clusters", type=int, default=5, help="Number of clusters to use for durations"
    )
    parser.add_argument(
        "train_text", type=str, help="Path to train text file. e.g. /home/user/data/train/text"
    )
    parser.add_argument(
        "train_f0", type=str, help="Path to train F0 file. e.g. /home/user/data/train/pitch"
    )
    parser.add_argument(
        "train_energy", type=str, help="Path to train energy file. e.g. /home/user/data/train/energy"
    )
    parser.add_argument(
        "train_f0_out", type=str, help="Path to output clustered F0 file. e.g. /home/user/data/train/pitch_clustered"
    )
    parser.add_argument(
        "train_energy_out", type=str, help="Path to output clustered energy file. e.g. /home/user/data/train/energy_clustered"
    )
    parser.add_argument(
        "valid_text", type=str, help="Path to validation text file /home/user/data/test/text"
    )
    parser.add_argument(
        "valid_f0", type=str, help="Path to validation F0 file /home/user/data/test/pitch"
    )
    parser.add_argument(
        "valid_energy", type=str, help="Output path for validation F0 file /home/user/data/test/energy"
    )
    parser.add_argument(
        "valid_f0_out", type=str, help="Output path for validation duration cluster IDs /home/user/data/test/cluster_ids"
    )
    parser.add_argument(
        "valid_energy_out", type=str, help="Output path for validation duration cluster IDs /home/user/data/test/cluster_ids"
    )
    parser.add_argument(
        "--f0min",
        type=int,
        default=70,
        help="Min frequency",
    )
    parser.add_argument(
        "--f0max",
        type=int,
        default=8000,
        help="Max frequency",
    )

    return parser

def read_transcript(path):
    for line in open(path, "r"):
        split = line.strip().split(" ")
        yield split[0], split[1:]

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

    train_f0 = []
    train_energy = []
    train_utt_ids = []
    train_phones = []

    valid_f0 = []
    valid_energy = []
    valid_utt_ids = []
    valid_phones = []

    f0_all = []
    # a map of phone->list of all (normalized) pitch values for that phone
    # this allows us to cluster per-phone
    phone_f0 = {}
    energy_all = []
    phone_energy = {}

    train_text = read_transcript(args.train_text)
    valid_text = read_transcript(args.valid_text)
    
    for (utt_id, f0), \
        (u2, energy), \
        (u3, phones) in zip(ReadHelper("scp:" + args.train_f0),
                            ReadHelper("scp:" + args.train_energy),
                            train_text):
        if utt_id != u2 or u2 != u3:
            raise Exception("MISMATCH")
        train_utt_ids += [utt_id]
        train_phones += [phones]
        train_f0 += [f0]
        f0_all += f0.tolist()
        train_energy += [energy]
        energy_all += energy.tolist()
        for i, phone in enumerate(phones):
            if phone not in phone_f0:
                phone_f0[phone] = []
            if phone not in phone_energy:
                phone_energy[phone] = []
            phone_f0[phone] += [f0[i]]
            phone_energy[phone] += [energy[i]]

    for (utt_id, f0), \
        (_, energy), \
        (_, phones) in zip(ReadHelper("scp:" + args.valid_f0), 
                            ReadHelper("scp:" + args.valid_energy),
                                valid_text):
        valid_utt_ids += [utt_id]
        valid_f0 += [f0]
        valid_energy += [energy]
        valid_phones += [phones]
        
    # normalize across all phones
    f0_all = np.array(f0_all)
    f0_mean = f0_all.mean(axis = 0)
    f0_stddev = f0_all.std(axis = 0)

    energy_all = np.array(energy_all)
    energy_mean = energy_all.mean(axis = 0)
    energy_stddev = energy_all.std(axis = 0)

    f0_all = [[(f - f0_mean) / f0_stddev] for f in f0_all]
    energy_all = [(e - energy_mean) / energy_stddev for e in energy_all]

    # cluster across all phones
    # f0_k=KMeans(args.num_clusters)
    # f0_k.fit(f0_all)

    # energy_k=KMeans(args.num_clusters)
    # energy_k.fit(energy_all)

    # f0_k_phones = {}
    # energy_k_phones = {}

    # now normalize & cluster per-phone
    for phone in phone_f0:
        phone_f0[phone] = [(f - f0_mean) / f0_stddev for f in phone_f0[phone]]
        phone_energy[phone] = [(e - energy_mean) / energy_stddev for e in phone_energy[phone]]

        # f0_k_phones[phone]=KMeans(args.num_clusters)
        # f0_k_phones[phone].fit(phone_f0[phone])

        # energy_k_phones[phone]=KMeans(args.num_clusters)
        # energy_k_phones[phone].fit(phone_energy[phone])

        # we don't use K-means as we want the order of the labels to be centered around a mean value

        # sort the list of durations for each phone and bucket into [num_clusters] s.t. each bucket contains an equal number of samples
        # produces a list of floats [num_clusters] long
        # each float corresponds to the high end of the cluster duration (and each index corresponds to the cluster label)
        # e.g. {
        #  "h":[10,15,27],
        #  "e":[20,45,57], (so cluster 0 <= 20, cluster 1 <= 45, etc)
        # etc
        # }
        # print(phone_durations)
        # phone_f0[phone] = np.unique(phone_f0[phone])
        # phone_energy[phone] = np.unique(phone_energy[phone])

        phone_f0[phone] = np.array(phone_f0[phone])
        phone_energy[phone] = np.array(phone_energy[phone])
        
        if len(phone_f0[phone]) >= args.num_clusters:
           phone_f0[phone] = phone_f0[phone][::len(phone_f0[phone]) // (args.num_clusters)][:args.num_clusters]
        else:
            phone_f0[phone] = np.pad(phone_f0[phone], ((args.num_clusters - phone_f0[phone].shape[0]) // 2), mode="symmetric")
        if len(phone_energy[phone]) >= args.num_clusters:
            phone_energy[phone] = phone_energy[phone][::len(phone_energy[phone]) // (args.num_clusters)][:args.num_clusters]
        else:
            phone_energy[phone] = np.pad(phone_energy[phone][-1], ((args.num_clusters - phone_energy[phone].shape[0]) // 2), mode="symmetric")
        
    # write train values
    with WriteHelper(f"ark,t,scp:{args.train_f0_out}.ark,{args.train_f0_out}.scp") as wh:
        for utt_id, f0, phones in zip(train_utt_ids, train_f0, train_phones):
            
            # f0 = (f0 - f0_mean) / f0_stddev
            # f0_clusters = f0_k.predict(np.expand_dims(f0,1))
            # f0_clusters = [ f0_k_phones[phone].predict((entry - f0_mean) / f0_stddev) for entry, phone in zip(f0, phones)]
            f0_clusters = []
            for val,phone in zip(f0, phones):
                indices = np.asarray(phone_f0[phone] >= val).nonzero()[0]
                if indices.any():
                    cluster_id = indices[0]
                else:
                    cluster_id = args.num_clusters - 1
                f0_clusters += [ cluster_id ]
            wh(utt_id, np.array(f0_clusters))
    
    with WriteHelper(f"ark,t,scp:{args.train_energy_out}.ark,{args.train_energy_out}.scp") as wh:
        for utt_id, energy, phones in zip(train_utt_ids, train_energy, train_phones):
            # energy = (energy - energy_mean) / energy_stddev
            # energy_clusters = energy_k.predict(energy)
            #  energy_clusters = [ energy_k_phones[phone].predict((entry - energy_mean) / energy_stddev) for entry, phone in zip(energy, phones)]
            energy_clusters = []
            for val,phone in zip(energy, phones):
                indices = np.asarray(phone_energy[phone] > val).nonzero()[0]
                if indices.any():
                    cluster_id = indices[0]
                else:
                    cluster_id = args.num_clusters - 1
                energy_clusters += [ cluster_id ]
            wh(utt_id, np.array(energy_clusters))
    
    with WriteHelper(f"ark,t,scp:{args.valid_f0_out}.ark,{args.valid_f0_out}.scp") as wh:
        for utt_id, f0, phones in zip(valid_utt_ids, valid_f0, valid_phones):
            # f0 = (f0 - f0_mean) / f0_stddev
            # f0_clusters = f0_k.predict(np.expand_dims(f0,1))
            # f0_clusters = [ f0_k_phones[phone].predict((entry - f0_mean) / f0_stddev) for entry, phone in zip(f0, phones)]
            f0_clusters = []
            for val,phone in zip(f0, phones):
                indices = np.asarray(phone_f0[phone] > val).nonzero()[0]
                if indices.any():
                    cluster_id = indices[0]
                else:
                    cluster_id = args.num_clusters - 1
                f0_clusters += [ cluster_id ]
            wh(utt_id, np.array(f0_clusters))
    
    with WriteHelper(f"ark,t,scp:{args.valid_energy_out}.ark,{args.valid_energy_out}.scp") as wh:
        for utt_id, energy, phones in zip(valid_utt_ids, valid_energy, valid_phones):
            # energy = (energy - energy_mean) / energy_stddev
            # energy_clusters = energy_k.predict(energy)
            # energy_clusters = [ energy_k_phones[phone].predict((entry - energy_mean) / energy_stddev) for entry, phone in zip(energy, phones)]
            energy_clusters = []
            for val,phone in zip(energy, phones):
                indices = np.asarray(phone_f0[phone] > val).nonzero()[0]
                if indices.any():
                    cluster_id = indices[0]
                else:
                    cluster_id = args.num_clusters - 1
                energy_clusters += [ cluster_id ]
            wh(utt_id, np.array(energy_clusters))

if __name__ == "__main__":
    main()
