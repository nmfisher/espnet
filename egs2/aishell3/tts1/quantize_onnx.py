import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'tts_model.onnx'
model_quant = 'tts_model.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
