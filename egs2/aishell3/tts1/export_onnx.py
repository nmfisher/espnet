#!/usr/bin/env python3
"""Convert TTS to ONNX
Using ESPnet.

Test command:

python convert_tts2onnx.py --tts-tag espnet/kan-bayashi_ljspeech_vits
"""
import argparse
import logging
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
    )

    # Prepare modules for conversion
    logging.info("Generate ONNX models")
    with torch.no_grad():
        
        model = tts.model.tts
        model.eval()

        # text: torch.Tensor,
        inputs = (
          torch.tensor([2, 15, 26, 39, 9,8, 10, 27, 3,11, 2],dtype=torch.int).to(device), 
          torch.tensor([1],dtype=torch.int).to(device), 
        )

        # plain method invocation to confirm that everything works correctly outside torch.jit.script
        model.forward = model.export

        model.forward(*inputs)
        
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
            input_names=['phones','speaker_id'],
            output_names=['pcm','durations'],
            dynamic_axes={
                'phones': {
                    0: 'length'
                },
                'pcm': {
                    0: 'olen', 
                },
                'durations': {
                    0: 'length'
                },
            }
        )
    print("Done!")