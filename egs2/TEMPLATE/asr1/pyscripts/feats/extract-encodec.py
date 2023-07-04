#!/usr/bin/env python3
import argparse
import logging
import numpy as np
from espnet.utils.cli_utils import get_commandline_args
from kaldiio import ReadHelper, WriteHelper
import torchaudio, torch
from encodec import EncodecModel
from encodec.utils import convert_audio

def get_parser():
    parser = argparse.ArgumentParser(
        description="Extracts quantized Encodec feats from raw waveform.",
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

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(1.5)
    lm = model.get_lm_model()
    
    with WriteHelper(args.outs) as writer:
        with ReadHelper(args.wavs) as reader:
            for utt_id, (fs, audio) in reader:
                audio = audio.astype(np.float32) / 32767
                if fs != 24000:
                    resampled = torchaudio.functional.resample(torch.Tensor(audio.copy()), fs, 24000)
                else:
                    resampled = audio
                with torch.no_grad():
                    encoded_frames = model.encode(resampled[None][None])
                    # frame = encoded_frames[0][0]
                    # _, K, T = frame.shape
                    # states = None
                    # offset = 0
                    # input_ = torch.zeros(1, K, 1, dtype=torch.long)
                    # output = []
                    # for t in range(T):
                    #     with torch.no_grad():
                    #         probas, states, offset = lm(input_, states, offset)
                    #     input_ = 1 + frame[:, :, t: t + 1]
                    #     output += [probas[0, :, :, 0].T.flatten()]
                    # output = np.vstack(output)
                    # output = np.concatenate([frame[0].T.float(), output],axis=1)
                    # writer(utt_id, output)

                    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
                    codes = codes.transpose(1,2)
                    writer(utt_id, codes[0].numpy().astype(np.float32))
                

if __name__ == "__main__":
    main()
