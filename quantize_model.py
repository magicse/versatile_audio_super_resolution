import os
import sys
import torch
from glob import glob
import soundfile as sf
import numpy as np
from LatentDiffusion_to_float16 import optimize_model_for_inference, quantize_model_inference  # or directly use the code

sys.path.append(os.path.join("audiosr"))

from audiosr.pipeline import super_resolution, build_model
from audiosr.utils import (
    instantiate_from_config,
    read_audio_file,
    save_wave,
    default_audioldm_config,
)

torch.set_float32_matmul_precision("high")

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = r"I:\AudioSr_model\Audiosr_basic\pytorch_model.bin"
save_path = r"I:\AudioSr_model\Audiosr_basic\clean_model\clean_basic.pth"

# Explicitly load weights without internet
checkpoint_path = os.path.abspath(os.path.join(model_path))

print("Loading model...")
audiosr = build_model(ckpt_path=checkpoint_path, model_name='basic', device=device)

# Load and optimize model for inference with safe quantization
# optimized_model = optimize_model_for_inference(audiosr, save_path=save_path, aggressive=False)

# If you already have a loaded model object, you can directly do:
optimized_model = quantize_model_inference(audiosr, aggressive=True)

# Switch to inference mode
optimized_model.eval()

# Disable gradients
for p in optimized_model.parameters():
    p.requires_grad = False

# Move to CUDA (if available)
if torch.cuda.is_available():
    optimized_model = optimized_model.to(device)
    torch.cuda.empty_cache()

# Save the optimized model
torch.save({
    'state_dict': optimized_model.state_dict(),
    'info': {
        'quantized': True,
        'dtype': 'float16',
        'aggressive': False
    }
}, save_path)

print(f"Model saved to {save_path}")
