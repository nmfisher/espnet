import sys
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

if len(sys.argv) < 2:
    print("Usage: quantize_onnx.py <input.onnx> <output.onnx>")
    sys.exit(-1)
model_fp32 = sys.argv[1] # 'tts_model.onnx'
model_quant = sys.argv[2] #'tts_model.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
