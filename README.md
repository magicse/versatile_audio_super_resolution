# Audio Upscaler (AudioSR)

[![arXiv](https://img.shields.io/badge/arXiv-2309.07314-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2309.07314)  [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://audioldm.github.io/audiosr)

## Overview

AudioSR is a powerful tool designed to enhance the fidelity of your audio files, regardless of their type (e.g., music, speech, ambient sounds) or sampling rates. It leverages cutting-edge super-resolution techniques to upscale audio signals, resulting in superior quality output.

## Key Features
- **Saves a clean**: quantized model ready for deployment (reduce size from 5.75Gb to 2.76Gb) file quantize_model.py
- **Force model path**: Force path model to load.
  
```
from audiosr.pipeline import build_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = os.path.join(r"I:\AudioSr_model\basic.pth")  
audiosr = build_model(ckpt_path=checkpoint_path, model_name='basic', device=device)
```

## Acknowledgments
Based on the work of https://github.com/haoheliu/versatile_audio_super_resolution/

```bibtex
@article{liu2023audiosr,
  title={{AudioSR}: Versatile Audio Super-resolution at Scale},
  author={Liu, Haohe and Chen, Ke and Tian, Qiao and Wang, Wenwu and Plumbley, Mark D},
  journal={arXiv preprint arXiv:2309.07314},
  year={2023}
}
```
