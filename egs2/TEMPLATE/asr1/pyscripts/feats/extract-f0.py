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
        "train_wav", type=str, help="Path to train WAV file. e.g. data/train/wav.scp"
    )
    parser.add_argument(
        "train_durations", type=str, help="Path to train durations file. e.g. data/train/durations"
    )
    parser.add_argument(
        "train_transcripts", type=str, help="Path to train transcripts. e.g. data/train/text"
    )
    parser.add_argument(
        "train_pitch_out", type=str, help="Output path for train pitch data/train/pitch"
    )
    parser.add_argument(
        "train_energy_out", type=str, help="Output path for train energy e.g. data/train/energy"
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
        "valid_pitch_out", type=str, help="Output path for validation pitch data/valid/pitch"
    )
    parser.add_argument(
        "valid_energy_out", type=str, help="Output path for validation pitch data/valid/pitch"
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
    parser.add_argument(
        "train_dump_dir", type=str, help="Directory where train pitch (as numpy arrays) will be written (one file per sample)"
    )
    parser.add_argument(
        "valid_dump_dir", type=str, help="Directory where validation pitch (as numpy arrays) will be written (one file per sample)"
    )
    return parser


def _open(path):
    with open(path, "r") as infile:
        for line in infile.readlines():
            split = line.strip().split(" ")
            yield (split[0], split[1:])


def extract(wavs, durations, transcripts, sample_rate, hop_length):

    stft = Stft(
          n_fft=1024,
          win_length=1024,
          hop_length=hop_length,
    )

    for (utt_id, path), (_, utt_durations), (_, transcript) in zip(wavs, durations, transcripts):

        utt_durations = [ float(x) for x in utt_durations ]

        wav, rate = librosa.load(path[0], sr=sample_rate)
        frames_per_second = sample_rate / hop_length

        f0, t = pyworld.dio(wav.astype(np.double), rate,
                            frame_period=frames_per_second)
        
        # Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, energy_lengths = stft(torch.tensor(np.expand_dims(wav.astype(np.double), 0)), torch.tensor([wav.shape[0]]))
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        energy = torch.sqrt(torch.clamp(input_power.sum(dim=2), min=1.0e-10))
        
        last = 0
        es = []
        for d in utt_durations:
          d = int(d)
          es += [ energy[0,last:last+d].sum() / d ]
          last += d
        es = np.vstack(es)
        utt_f0 = []

        # convert durations from frames to millisecond times
        times = [x / frames_per_second for x in utt_durations]
        
        # change the phone durations into start/end timestamps such that
        # (start_time,end_time) for phone[i] is (phone_times[i-1],phone_times[i])
        phone_times = reduce(lambda accum, x: accum
                             + [x + accum[-1]], times, [0])
        start = 0
        # iterate over every phone timestamp
        for time, phone in zip(phone_times[1:], transcript):
            # iterate from the last F0 timestamp until we find a time that's higher (i.e. later) than the phone timestamp
            accum = 0
            num_frames = 0
            for i in range(start, f0.shape[0], 1):
              accum += f0[i]
              num_frames += 1
              if t[i] >= time  or i == f0.shape[0] - 1:
                  break
            # add the F0 value (or simple average, if it spanned more than 1 frame)
            utt_f0.append(accum / num_frames)
            start = i
                  
        if len(utt_f0) != len(transcript):
            print(f"{path}\n transcript : {transcript}\n f0 : {utt_f0}\n DIO timestamps {t}\n phone_times {phone_times}\n original durations: {utt_durations}")
            raise Exception("length mismatch")
                
        yield (utt_id, utt_f0,es)


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
    train_vals = list(extract(_open(args.train_wav), _open(
        args.train_durations), _open(args.train_transcripts), args.sample_rate, args.hop_length))

    valid_vals = extract(_open(args.valid_wav), _open(
        args.valid_durations), _open(args.valid_transcripts), args.sample_rate, args.hop_length)

    # normalize train/valid pitches based on mean/stddev from train only
    # todo - multi-speaker basis
    f0=np.array(list(itertools.chain.from_iterable([f for utt_id,f,e in train_vals])))
    
    mean_f0 = f0.mean(axis = 0)
    stddev_f0 =f0.std(axis = 0)

    energy =np.array(list(itertools.chain.from_iterable([e for utt_id,f,e in train_vals])))
    mean_e = energy.mean(axis = 0)
    stddev_e = energy.std(axis=0)

    # normalize training/validation set based on training means/stddevs
    train_vals = [ (utt_id, (f0 - mean_f0) / stddev_f0, (energy - mean_e) / stddev_e)  for (utt_id, f0, energy) in train_vals]
    valid_vals = [ (utt_id, (f0 - mean_f0) / stddev_f0, (energy - mean_e) / stddev_e) for (utt_id, f0, energy) in valid_vals]
    
    for vals, pitch_out, energy_out, dump_dir in [(train_vals, args.train_pitch_out, args.train_energy_out, args.train_dump_dir), (valid_vals, args.valid_pitch_out,args.valid_energy_out, args.valid_dump_dir)]:
        # dump the actual F0 values to a .npy file
        # write a line in Kaldi scp format pointing to this file
        with open(pitch_out, "w") as pitch_scp,open(energy_out, "w") as energy_scp:
            for utt_id, f0, energy in vals:
                pitch_path = dump_dir + f"/{utt_id}_pitch.npy"
                energy_path = dump_dir + f"/{utt_id}_energy.npy"
                np.save(pitch_path,np.expand_dims(f0,1),allow_pickle=False)
                print(energy)
                np.save(energy_path,energy,allow_pickle=False)
                pitch_scp.write("{} {}\n".format(utt_id, pitch_path))
                energy_scp.write("{} {}\n".format(utt_id, energy_path))

if __name__ == "__main__":
    main()
