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
        help="sample rate for the input adio"
    )
    parser.add_argument(
        "hop_length",
        type=int,
        default=256,
        help="hop_length used during feat extraction (this is needed as we perform a STFT to extract pitch/energy. This should match either the hop length used during phone alignment or BFCC extraction"
    )
    parser.add_argument(
        "wav", type=str, help="Path to train WAV file. e.g. data/train/wav.scp"
    )
    parser.add_argument(
        "durations", type=str, help="Path to train durations file. e.g. data/train/durations"
    )
    parser.add_argument(
        "transcripts", type=str, help="Path to train transcripts. e.g. data/train/text"
    )
    parser.add_argument(
        "pitch_out", type=str, help="Output path for train pitch data/train/pitch"
    )
    parser.add_argument(
        "energy_out", type=str, help="Output path for train energy e.g. data/train/energy"
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
        "dump_dir", type=str, help="Directory where train pitch (as numpy arrays) will be written (one file per sample)"
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


def extract(wavs, durations, transcripts, sample_rate, hop_length):

    stft = Stft(
          n_fft=1024,
          win_length=1024,
          hop_length=hop_length,
    )

    for (utt_id, path), (utt_id2, utt_durations), (utt_id3, transcript) in zip(wavs, durations, transcripts):

        if utt_id2 != utt_id or utt_id3 != utt_id:
          raise Exception(f"Utterance ID mismatch {utt_id} vs {utt_id2} vs {utt_id3}, are the wav.scp/text/durations files sorted and contain the exact same utterances in the exact same order?")
        if len(utt_durations) != len(transcript):
          print(utt_durations)
          print(transcript)
          raise Exception(f"Number of phone frame durations {len(utt_durations)} does not match number of phones in transcript {len(transcript)} for utt {utt_id}, do the wav.scp/text/durations files all match exactly?")

        utt_durations = [ int(x) for x in utt_durations ]

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
            # print(f"{path}\n transcript : {transcript}\n f0 : {utt_f0}\n DIO timestamps {t}\n phone_times {phone_times}\n original durations: {utt_durations}")
            raise Exception(f"length mismatch for {utt_id} : {len(utt_f0)} vs {len(transcript)}")
        print(f"Extracted F0 and energy for utterance {utt_id}")
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
    vals = list(extract(_open(args.wav), _open(
        args.durations), _open(args.transcripts), args.sample_rate, args.hop_length))
    
    # dump the actual F0 values to a .npy file
    # write a line in Kaldi scp format pointing to this file
    with open(args.pitch_out, "w") as pitch_scp,open(args.energy_out, "w") as energy_scp:
        for utt_id, f0, energy in vals:
            pitch_path = args.dump_dir + f"/{utt_id}_pitch.npy"
            energy_path = args.dump_dir + f"/{utt_id}_energy.npy"
            np.save(pitch_path,np.expand_dims(f0,1),allow_pickle=False)
            np.save(energy_path,energy,allow_pickle=False)
            pitch_scp.write("{} {}\n".format(utt_id, pitch_path))
            energy_scp.write("{} {}\n".format(utt_id, energy_path))

if __name__ == "__main__":
    main()
