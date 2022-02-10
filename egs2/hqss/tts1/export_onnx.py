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
from espnet2.tts.hqss import HQSS

import torch.nn.functional as F


def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tts-tag",
        required=True,
        type=str,
        help="TTS tag (or Directory) for model located at huggingface/zenodo/local"
    )
    parser.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
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

    model = Text2Speech.from_pretrained(
        model_file="./exp/tts_train_hqss_raw_word/train.loss.best.pth",
        train_config="./exp/tts_train_hqss_raw_word/config.yaml",
        vocoder_tag=None,
        device="cuda",
    )

    # Prepare modules for conversion
    logging.info("Generate ONNX models")
    with torch.no_grad():
        device = "cpu"#text2speech.device
        # preprocessing = text2speech.preprocess_fn
        # model_tts = text2speech.tts
        # model = HQSS(20,20,zoneout_rate=0.0)
        model.eval()

        model.forward = model.export

        # scripted_module = torch.jit.trace(model)
        
        # Generate TTS Model
        torch.onnx.export(
            #scripted_module,
            model,
            # model_tts.to(device),
            (
              torch.zeros(10,dtype=torch.int).to(device), 
              torch.zeros(10,dtype=torch.int).to(device),
              torch.zeros(10,dtype=torch.int).to(device),
              torch.zeros(1,dtype=torch.int).to(device)
            ),
            
            'tts_model.onnx',
            # example_outputs=torch.zeros(100,20),
            export_params=True,
            opset_version=13,
            do_constant_folding=False,
            verbose=True,
            input_names=['phones', 'durations', 'pitch', 'sids' ],
            output_names=['wav'],
            dynamic_axes={
                'phones': {
                    0: 'length'
                },
                'durations': {
                    0: 'length'
                },
                'pitch': {
                    0: 'length'
                },
                'wav': {
                    0: 'length'
                }
            }
        )
    print("Done!")