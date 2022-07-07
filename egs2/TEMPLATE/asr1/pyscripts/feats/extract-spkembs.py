#!/usr/bin/env python3.7
# from byol_a.common import *
# from byol_a.augmentations import PrecomputedNorm
# from byol_a.models import AudioNTT2020


import os

import argparse
import logging
import sys
import numpy as np
from espnet.utils.cli_utils import get_commandline_args

import scipy.io.wavfile
from kaldiio import ReadHelper, WriteHelper
import subprocess
import numpy as np
from sklearn.decomposition import PCA
import torch 
import random

def get_parser():
    parser = argparse.ArgumentParser(
        description="computes (averaged) speaker embeddings for dataset using pretrained BYOL-A.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "wavs", type=str, help="Path to wav.scp (no scp prefix needed)"
    )
    parser.add_argument(
        "outs", type=str, help="wspecifier for test outputs (e.g. 'ark,scp:xvectors.ark,xvector.scp')"
    )
    # parser.add_argument(
    #     "train_wavs", type=str, help="Path to wav.scp (no scp prefix needed)"
    # )
    # parser.add_argument(
    #     "test_wavs", type=str, help="Path to wav.scp (no scp prefix needed)"
    # )
    # parser.add_argument(
    #     "train_outs", type=str, help="wspecifier for outputs (e.g. 'ark,scp:xvectors.ark,xvector.scp')"
    # )
    # parser.add_argument(
    #     "test_outs", type=str, help="wspecifier for test outputs (e.g. 'ark,scp:xvectors.ark,xvector.scp')"
    # )
    # parser.add_argument(
    #     "stats", type=str, help="path where stats (melspec mean/stddev) will be written.", 
    # )
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

    device = torch.device('cuda')
    # cfg = load_yaml_config('/home/hydroxide/projects/byol-a/config.yaml')

    # # Preprocessor and normalizer.
    # to_melspec = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=cfg.sample_rate,
    #     n_fft=cfg.n_fft,
    #     win_length=cfg.win_length,
    #     hop_length=cfg.hop_length,
    #     n_mels=cfg.n_mels,
    #     f_min=cfg.f_min,
    #     f_max=cfg.f_max,
    # )

    # vals=[]
    # if args.stats is None:
    #     raise Exception("Stats path must be provided")
    # print(f"Calculating stats, will be written to {args.stats}")
    # # first pass through to calculate mean and stddev
    # count=0
    # for line in open(args.train_wavs).readlines():
    #     split=line.strip().split(" ")
    #     utt_id = split[0]
        
    #     wav, sr = torchaudio.load(split[1]) # a sample from SPCV2 for now
    #     if sr != cfg.sample_rate:
    #         raise Exception("Expected sample rate %d but %d was provided for %s, please make sure all audio inputs are resampled first." % (cfg.sample_rate, sr, utt_id))
    #     melspec = to_melspec(wav).reshape(-1)
        
    #     vals += [ (melspec + torch.finfo(torch.float).eps).log() ]
    # all = np.concatenate(vals)
    # stats = np.array([all.mean(), all.std()])
    # stats.tofile(args.stats)
    # print(f"Wrote stats to {args.stats}")
    
    # Mean and standard deviation of the log-mel spectrogram of input audio samples, pre-computed.
    # See calc_norm_stats in evaluate.py for your reference.
    # Load pretrained weights.
    # model = AudioNTT2020(d=cfg.feature_d)
    # model.load_weight('/home/hydroxide/projects/byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth', device)
    # model.eval()
    
    # normalizer = PrecomputedNorm(stats)

    # from speechbrain.pretrained import EncoderClassifier
    # classifier = EncoderClassifier.from_hparams(source="LanceaKing/spkrec-ecapa-cnceleb",run_opts={"device":"cuda"})

    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    print(nemo_asr.models.EncDecSpeakerLabelModel.list_available_models())
    speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='speakerverification_speakernet')
    speaker_model = speaker_model.to(device)
    speaker_model.eval()
    try:
        from torch.cuda.amp import autocast
    except ImportError:
        from contextlib import contextmanager


    def vectorize(wavs):
        utt_ids =[]
        vecs = []
        spk2sid={}
        spkrs=[]
        i = 0
        with ReadHelper(wavs) as utts:
            for utt_id, (rate, audio) in utts:
                # for line in open(wavs).readlines():
                    # split=line.strip().split(" ")
                    # utt_id = split[0]

                if rate != 16000:
                    raise Exception("Incorrect sample rate")

                utt_ids.append(utt_id)
                spkr=utt_id.split("-")[0]

                if spkr not in spk2sid:
                  spk2sid[spkr] = len(spk2sid)
                
                spkrs += [spk2sid[spkr]]
                    
                    # wav, sr = torchaudio.load(split[1]) # a sample from SPCV2 for now
                    # if rate != cfg.sample_rate:
                    #     raise Exception("Expected sample rate %d but %d was provided for %s, please make sure all audio inputs are resampled first." % (cfg.sample_rate, sr, utt_id))
        #     melspec = to_melspec(wav)
        # #     # Convert to a log-mel spectrogram, then normalize.
        #     lms = normalizer((melspec + torch.finfo(torch.float).eps).log())

        #     # Now, convert the audio to the representation.
        #     vec = model(lms.unsqueeze(0)).cpu().detach().numpy()
                # vec = classifier.encode_batch(torch.tensor(audio))
                # raise Exception(audio.shape)
                input_signal = torch.tensor(audio.astype(np.float32)).unsqueeze(0).to(device)
                input_signal_length = torch.tensor([audio.shape[0]]).to(device)
                # print(input_signal.size())
                # print(input_signal_length.size())
                _, vec = speaker_model.forward(input_signal=input_signal, input_signal_length=input_signal_length)
                
                # print("Encoded")
                vecs.append(vec.detach().cpu().numpy())

                i += 1
                # if i > 256:
                #   break
        # vecs =np.concatenate(vecs, 0).squeeze()
        
        # transformed = TSNE().fit_transform(vecs)
        # scatter = plt.scatter(transformed[:,0], transformed[:,1], c=spkrs)
        # handles, _ = scatter.legend_elements(prop='colors')
        # import random;
        # r=random.randint(0,1000)
        # plt.savefig(f"myplot_{r}.png")

        # plt.legend(handles, spkrs)

        return utt_ids, vecs

    def average(utt_ids, vecs):
        spkr_vectors={}
        for i in range(len(utt_ids)):
            utt_id=utt_ids[i]
            spkr=utt_id.split("-")[0]
            if spkr not in spkr_vectors:
                spkr_vectors[spkr] = {
                    "count":1,
                    "vector":vecs[i]
                }
            else:
                spkr_vectors[spkr]["vector"] += vecs[i]
                spkr_vectors[spkr]["count"] += 1
        for spkr in spkr_vectors.keys():
            spkr_vectors[spkr]["vector"] / spkr_vectors[spkr]["count"]
        return spkr_vectors

    def shuffle_by_speaker(utt_ids, vecs):
        spkr_vectors={}
        for i in range(len(utt_ids)):
            utt_id=utt_ids[i]
            spkr=utt_id.split("-")[0]
            if spkr not in spkr_vectors:
                spkr_vectors[spkr] = []

            spkr_vectors[spkr] += [i]

        for spkr in spkr_vectors.keys():
            random.shuffle(spkr_vectors[spkr])
            # random.shuffle(spkr_vectors[spkr])
            # random.shuffle(spkr_vectors[spkr])
            # random.shuffle(spkr_vectors[spkr])
            # random.shuffle(spkr_vectors[spkr])

        misses = 0

        iterators = {spkr:iter(spkr_vectors[spkr]) for spkr in spkr_vectors.keys()}
        for i in range(len(utt_ids)):
            utt_id = utt_ids[i]
            spkr=utt_id.split("-")[0]
            shuffled_index = next(iterators[spkr])

            if shuffled_index == i:
                misses += 1
            yield vecs[shuffled_index]
        
        print(f"Misses : {misses}")
    train_utt_ids, train_vecs  = vectorize(args.wavs)
    print(f"Vectorized {len(train_utt_ids)} wavs")
    
    # test_utt_ids, test_vecs  = vectorize(args.test_wavs)

    # pca=PCA(n_components=64)
    # train_vecs = pca.fit_transform(train_vecs)
    # test_vecs = pca.transform(test_vecs)

    # train_vecs = average(train_utt_ids, train_vecs)
    # test_vecs = average(test_utt_ids, test_vecs)

    train_vecs = list(shuffle_by_speaker(train_utt_ids, train_vecs))
    
    with WriteHelper(args.outs) as writer:
        for i in range(len(train_utt_ids)):
            utt_id = train_utt_ids[i]
            # writer(utt_id, train_vecs[utt_id.split("-")[0]]["vector"])
            writer(utt_id, train_vecs[i])
    print(f"Wrote {len(train_utt_ids)} vectors")
    # with WriteHelper(args.test_outs) as writer:
    #     for i in range(len(test_utt_ids)):
    #         utt_id = test_utt_ids[i]
    #         writer(utt_id, test_vecs[utt_id.split("-")[0]]["vector"])
        
if __name__ == "__main__":
    main()
