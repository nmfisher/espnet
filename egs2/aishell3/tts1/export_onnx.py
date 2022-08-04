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
from espnet2.utils.types import str_or_none

import torch.nn.functional as F

def get_parser():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser

def prune(model):
    return model #TODO
    #     import torch.nn.utils.prune as prune
    #     handled = []
    #     modules = [model]
    #     params = []
    #     while len(modules) > 0:
    #         module = modules.pop()

    #         if module in handled:
    #             print(f"{name} already handled, skipping")
    #             continue

    #         if len(list(module.named_children())) == 0:
    #             for name, param in module.named_parameters():
    #                 if name in ['bias']:
    #                     params += [ (module, 'bias') ]
    #                 elif name in ['weight']:
    #                     params += [ (module, 'weight') ]
    #         else:
    #             modules += list([child for name, child in module.named_children()])
    #         handled += [ module ] 
            
    #     if len(params) > 0:
    #         print(f"Params for pruning : {params}")
    #         # raise Exception()
    #         prune.global_unstructured(
    #             params,
    #             pruning_method=prune.L1Unstructured,
    #             amount=0.7,
    #         )
    #         for module, param in params:
    #             prune.remove(module, param)

if __name__ == "__main__":
    # Logger
    parser = get_parser()

    parser.add_argument(
        "model", type=str, help="Path to checkpoint .pth file")

    parser.add_argument(
        "config", type=str, help="Path to config .yaml")

    parser.add_argument(
        "output", type=str, help="Path to output .onnx file"
    )

    parser.add_argument(
        "--prune", help="Whether or not to prune the model", action="store_true"
    )

    args = parser.parse_args()

    if args.model is None or args.config is None or args.output is None:
        sys.exit()

    # Load Pretrained model and testing wav generation
    device = "cpu"
    tts = Text2Speech.from_pretrained(
        model_file=args.model, # "./exp/tts_train_wlsc_wlsc_phn_none_student_ok/latest.pth",
        train_config=args.config, # "./exp/tts_train_wlsc_wlsc_phn_none_student_ok/inference.yaml",
        vocoder_tag=None,
        device=device,
        num_speakers=9
    )

    # # Prepare modules for conversion
    with torch.no_grad():
        
        model = tts.model.tts
        if args.prune:
            model = prune(model)
        
        model.eval()
        text = torch.tensor([45, 34, 34, 34, 34, 101],dtype=torch.long).to(device)
        phone_word_mappings = torch.tensor([0,0,0,1,1,1])
        feats_avg = torch.randn(2,20).to(device) # fake BFCCs
        sids = torch.tensor([3]).to(device) # speaker IDs
        spembs = torch.randn(1,256)

        # plain method invocation to confirm that everything works correctly outside torch.jit.script
        odict = model.export(text, sids=sids, phone_word_mappings=phone_word_mappings,feats_word_avg=feats_avg, spembs=spembs)
        
        inputs = (text,sids,phone_word_mappings,feats_avg,spembs)

        model.forward = model.export
        
        # now try conversion to TorchScript
        scripted_module = torch.jit.script(model, inputs)
        
        # export
        torch.onnx.export(
            scripted_module,
            inputs,
            args.output,
            export_params=True,
            opset_version=13,
            do_constant_folding=False,
            verbose=True,
            input_names=[
                'phones', 'speaker_id', 'phone_word_mappings','style_reference','spembs' 
            ],
            output_names=['pcm','durations'],
            dynamic_axes={
                'phones': {
                    0: 'length'
                },
                'style_reference': {
                    0: 'length'
                },
                'phone_word_mappings':{
                    0: 'num_phones'
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

        
