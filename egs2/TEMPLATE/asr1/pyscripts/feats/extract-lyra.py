#!/usr/bin/env python3.7
import argparse
import logging
import sys
import numpy as np
from espnet.utils.cli_utils import get_commandline_args
import os
import scipy.io.wavfile
from spafe.features.bfcc import bfcc
from kaldiio import ReadHelper, WriteHelper
import subprocess
import tempfile
import soundfile as sf
import tensorflow as tf

def get_parser():
    parser = argparse.ArgumentParser(
        description="Extracts quantized Lyra feats from raw waveform.",
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

    output_dir = tempfile.mkdtemp()
    model_path = os.environ["LYRA_MODEL_PATH"]
    if model_path is None or model_path == "":
        raise Exception("Environment variable LYRA_MODEL_PATH must be provided")

    with WriteHelper(args.outs) as writer:
        with ReadHelper(args.wavs) as reader:
            for utt_id, (fs, audio) in reader:
                if fs != 16000:
                    raise Exception("Expected 16000Hz PCM16 audio")
                wav_path = os.path.join(output_dir, f"{utt_id}.wav")
                sf.write(wav_path, audio, fs)
                encoder = tf.lite.Interpreter("/home/hydroxide/projects/lyra/lyra/model_coeffs/soundstream_encoder.tflite")
                encoder.allocate_tensors()

                encoder_runner = encoder.get_signature_runner("serving_default")
                encoder_input_details = encoder_runner.get_input_details()
                encoder_outputs = []
                audio = audio.astype(np.float32) / 32767
                for i in range(0, audio.shape[0], 320):
                    audio_segment = audio[i:i+320]
                    if audio_segment.shape[0] < 320:
                        audio_segment = np.pad(audio_segment, (0,320 - audio_segment.shape[0]))
                    output = encoder_runner(input_audio=audio_segment)
                    encoder_outputs += [ output["output_0"] ]
                feats = np.array(encoder_outputs).squeeze()

                # process = subprocess.run(['encoder_main', "--input_path", wav_path, "--output_dir", output_dir, "--bitrate", "6000", "--model_path", model_path ])
                # lyra_out = wav_path.replace(".wav", ".lyra")
                # #print(f"Extracted Lyra feats from {wav_path} to {lyra_out}")
                # int8=[]
                # with open(lyra_out, "rb") as infile:
                #     for i in infile.read():
                #         v1 = i & 0b11110000
                #         v2 = i & 0b00001111
                #         int8 += [v1 >> 4, v2]
                # feats = np.array(int8).astype(np.int32)
                # feats = feats.reshape(-1, 30).astype(np.float32) # even though these are integers, espnet/kaldiio only allows matrices to be stored as float32 so we convert 
                # this avoids needing to constantly reshape later in the pipeline


                writer(utt_id, feats)
#                print(f"Wrote Lyra feats of length {feats.shape} to file {lyra_out}")
                

if __name__ == "__main__":
    main()
