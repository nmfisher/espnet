#!/usr/bin/env python3
"""Convert TTS to ONNX
Using ESPnet.

Test command:

python convert_tts2onnx.py --tts-tag espnet/kan-bayashi_ljspeech_vits
"""
import argparse
import logging
from re import X
import sys
import numpy as np
import torch
import time

from typing import Dict
from typing import Optional

from espnet2.bin.tts_inference import Text2Speech
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none

import torch.nn.functional as F

class FooModel(torch.nn.Module):
    def forward(self, x):
        return x, x
def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser

if __name__ == "__main__":
    # Logger
    parser = get_parser()
    args = parser.parse_args()

    #logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    # logging.basicConfig(filename='onnx.log', encoding='utf-8', level=logging.INFO, format=logfmt)

    # Load Pretrained model and testing wav generation
    #logging.info("Preparing pretrained model from: %s", args.tts_tag)
    device = "cpu"#text2speech.device
    tts = Text2Speech.from_pretrained(
        model_file="./exp/tts_train_gst+xvector_conformer_fastspeech2_bfcc_phn_none/train.loss.best.pth",
        train_config="./exp/tts_train_gst+xvector_conformer_fastspeech2_bfcc_phn_none/config.yaml",
        vocoder_tag=None,
        device=device,
        num_speakers=9
    )

    # Prepare modules for conversion
    logging.info("Generate ONNX models")
    with torch.no_grad():
        
        model = tts.model.tts
        
        model.eval()
        #import numpy as np
        text = torch.tensor([140,130,69,131,115,144,99,18,99,14],dtype=torch.int).to(device)
        # with open("/tmp/tmp.feats.txt") as infile:
        #     lines = infile.readlines()
        #     data =[]
        #     for line in lines[1:]:
        #         line = line.strip().split(" ")
        #         line = [l for l in line  if len(l) > 0 and l != "]"]
        #         data += [[float(x) for x in line]]
        # feats = np.fromfile("exp/tts_train_gst+xvector_conformer_fastspeech2_bfcc_phn_none/inference_train.loss.best/test/log/output.28/norm/zhCNXiaoyouNeural-d270e74e19ca5b9a8ff86f4d80214dfa_duration_1.1.npy",dtype=np.float32).reshape(-1,20)
        # feats = torch.tensor(feats).to(device)
        feats = torch.randn(150,20).to(device) # fake BFCCs
        sids = torch.tensor([4]).to(device) # speaker IDs

        odict = model.inference(text, feats=feats, sids=sids)
        odict["feat_gen"].detach().numpy().tofile("/tmp/torch_bfccs")
        
        inputs = (text, feats, sids)

        # plain method invocation to confirm that everything works correctly outside torch.jit.script
        model.forward = model.export

        #bfccs.detach().numpy().tofile("/tmp/torch_bfccs")
        
        # now try converstion to TorchScript
        scripted_module = torch.jit.script(model, inputs)
        
        # export
        torch.onnx.export(
            scripted_module,
            # model,
            inputs,
            'tts_model.onnx',
            export_params=True,
            opset_version=14,
            do_constant_folding=False,
            verbose=True,
            input_names=[
                'phones', 'style_reference', 'speaker_id'
            ],
            output_names=['pcm','durations'],
            dynamic_axes={
                'phones': {
                    0: 'length'
                },
                'style_reference': {
                    0: 'length'
                },
                'pcm': {
                    0: 'olen', 
                },
                'durations': {
                    1: 'length'
                },
            }
        )
    print("Done!")
