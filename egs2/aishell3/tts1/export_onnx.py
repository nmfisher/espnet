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

    # Load Pretrained model and testing wav generation
    device = "cpu"#text2speech.device
    tts = Text2Speech.from_pretrained(
        # model_file="./exp/tts_train_wlsc_wlsc_phn_none_student/latest.pth",
        # train_config="./exp/tts_train_wlsc_wlsc_phn_none_student/inference.yaml",
        model_file="./exp/tts_train_wlsc_wlsc_phn_none/latest.pth",
        train_config="./exp/tts_train_wlsc_wlsc_phn_none/config.yaml",
        vocoder_tag=None,
        device=device,
        num_speakers=9
    )

    # # Prepare modules for conversion
    # logging.info("Generate ONNX models")
    with torch.no_grad():
        
        model = tts.model.tts
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
        
        model.eval()
        #import numpy as np
        text = torch.tensor([154,  22,  80,  19,  69,  18,  57,  32, 132,  71,  68,  14,  97, 154],dtype=torch.int).to(device)
        phone_word_mappings = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6])
        feats = torch.randn(150,20).to(device) # fake BFCCs
        feats_avg = torch.randn(7,20).to(device) # fake BFCCs
        sids = torch.tensor([3]).to(device) # speaker IDs

        # plain method invocation to confirm that everything works correctly outside torch.jit.script
        odict = model.export(text, sids=sids, phone_word_mappings=phone_word_mappings,feats_word_avg=feats_avg)
        
        odict[0].detach().numpy().tofile("/tmp/torch_bfccs")

        print(odict[0])
        
        # odict["feat_gen"].detach().numpy().tofile("/tmp/torch_bfccs")
        
        inputs = (text, sids,phone_word_mappings,feats_avg)

        model.forward = model.export
        
        # now try conversion to TorchScript
        scripted_module = torch.jit.script(model, inputs)
        
        # export
        torch.onnx.export(
            scripted_module,
            #model,
            inputs,
            'tts_model.onnx',
            export_params=True,
            opset_version=14,
            do_constant_folding=False,
            verbose=True,
            input_names=[
                'phones', 'speaker_id', 'phone_word_mappings','style_reference', 
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

        # with open("/tmp/tmp.feats.txt") as infile:
        #     lines = infile.readlines()
        #     data =[]
        #     for line in lines[1:]:
        #         line = line.strip().split(" ")
        #         line = [l for l in line  if len(l) > 0 and l != "]"]
        #         data += [[float(x) for x in line]]
        # feats = np.fromfile("exp/tts_train_gst+xvector_conformer_fastspeech2_bfcc_phn_none/inference_train.loss.best/test/log/output.28/norm/zhCNXiaoyouNeural-d270e74e19ca5b9a8ff86f4d80214dfa_duration_1.1.npy",dtype=np.float32).reshape(-1,20)
        # feats = torch.tensor(feats).to(device)