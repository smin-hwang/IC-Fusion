import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.core import YAMLConfig

yaml_path = "configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
cfg = YAMLConfig(yaml_path)
model = cfg.model
model.eval()

input_shape = (1, 3, 640, 640)
x_rgb = torch.randn(input_shape)
x_ir = torch.randn(input_shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
x_rgb = x_rgb.to(device)
x_ir = x_ir.to(device)

flops = FlopCountAnalysis(model, (x_rgb, x_ir))
print("✅ FLOPs:", flops.total() / 1e9, "GFLOPs")
print("✅ Params:\n", parameter_count_table(model))
