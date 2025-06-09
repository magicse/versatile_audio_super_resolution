import torch
import torch.nn as nn
from typing import Union, Dict, Any
import logging

class Float16Quantizer:
    """Class for quantizing LatentDiffusion model to float16"""
    
    def __init__(self, model: 'LatentDiffusion'):
        self.model = model
        self.original_dtype = next(model.parameters()).dtype
        self.device = next(model.parameters()).device
        
    def quantize_to_float16(self, 
                           keep_first_stage_fp32: bool = True,
                           keep_cond_stage_fp32: bool = False) -> 'LatentDiffusion':
        """
        Quantize the model to float16
        
        Args:
            keep_first_stage_fp32: Keep first_stage_model in fp32 for stability
            keep_cond_stage_fp32: Keep cond_stage_models in fp32
        """
        logging.info("Starting quantization to float16...")
        
        # Always quantize the main diffusion model
        if hasattr(self.model, 'model'):
            self.model.model = self.model.model.half()
            logging.info("Diffusion model quantized to float16")
        
        # First stage model (VAE/Autoencoder) — optionally stay in fp32
        if hasattr(self.model, 'first_stage_model'):
            if keep_first_stage_fp32:
                logging.info("First stage model remains in fp32 for stability")
            else:
                self.model.first_stage_model = self.model.first_stage_model.half()
                logging.info("First stage model quantized to float16")
        
        # Conditional stage models
        if hasattr(self.model, 'cond_stage_models'):
            if keep_cond_stage_fp32:
                logging.info("Conditional stage models remain in fp32")
            else:
                for i, cond_model in enumerate(self.model.cond_stage_models):
                    self.model.cond_stage_models[i] = cond_model.half()
                logging.info("Conditional stage models quantized to float16")
        
        # Quantize model buffers
        self._quantize_buffers()
        
        logging.info("Quantization successfully completed")
        return self.model
    
    def _quantize_buffers(self):
        """Quantize model buffers"""
        for name, buffer in self.model.named_buffers():
            if buffer.dtype == torch.float32:
                # Keep numerically sensitive buffers in fp32
                if any(keyword in name.lower() for keyword in ['logvar', 'beta', 'alpha']):
                    continue
                # Quantize remaining buffers
                setattr(self.model, name.split('.')[-1], buffer.half())
    
    def smart_quantize(self) -> 'LatentDiffusion':
        """
        Smart quantization with critical components kept in fp32
        """
        logging.info("Starting smart quantization...")
        
        # Main UNet model — safe to quantize
        if hasattr(self.model, 'model'):
            self.model.model = self.model.model.half()
        
        # VAE encoder/decoder — important for quality, keep in fp32
        if hasattr(self.model, 'first_stage_model'):
            logging.info("Keeping first_stage_model in fp32 for quality")
        
        # Conditional models — depends on model type
        if hasattr(self.model, 'cond_stage_models'):
            for i, cond_model in enumerate(self.model.cond_stage_models):
                # Models like CLIP, BERT can be safely quantized
                if any(model_type in str(type(cond_model)) for model_type in 
                       ['CLIP', 'BERT', 'T5', 'FrozenCLIP']):
                    self.model.cond_stage_models[i] = cond_model.half()
                    logging.info(f"Conditional model {i} quantized to float16")
        
        return self.model


def quantize_model_inference(model: 'LatentDiffusion', 
                           aggressive: bool = False) -> 'LatentDiffusion':
    """
    Quick function to quantize a model for inference
    
    Args:
        model: LatentDiffusion model
        aggressive: Apply aggressive quantization (including VAE)
    """
    model.eval()  # Set to inference mode
    
    with torch.no_grad():
        quantizer = Float16Quantizer(model)
        
        if aggressive:
            # Quantize all components
            model = quantizer.quantize_to_float16(
                keep_first_stage_fp32=False,
                keep_cond_stage_fp32=False
            )
        else:
            # Safe quantization
            model = quantizer.smart_quantize()
    
    return model


def create_inference_wrapper(model: 'LatentDiffusion') -> 'InferenceWrapper':
    """Create a wrapper for optimized inference"""
    
    class InferenceWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.autocast_enabled = True
            
        @torch.no_grad()
        def generate_batch(self, *args, **kwargs):
            """Generation with automatic mixed precision"""
            if self.autocast_enabled and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    return self.model.generate_batch(*args, **kwargs)
            else:
                return self.model.generate_batch(*args, **kwargs)
        
        @torch.no_grad()
        def sample(self, *args, **kwargs):
            """Sampling with optimization"""
            if self.autocast_enabled and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    return self.model.sample(*args, **kwargs)
            else:
                return self.model.sample(*args, **kwargs)
        
        def __getattr__(self, name):
            """Proxy attribute access to the original model"""
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.model, name)
    
    return InferenceWrapper(model)


# Example usage:
def optimize_model_for_inference(model_path: str, 
                                save_path: str = None,
                                aggressive: bool = False) -> 'LatentDiffusion':
    """
    Full optimization pipeline for inference
    
    Args:
        model_path: Path to the model file or model instance
        save_path: Optional path to save the optimized model
        aggressive: Whether to apply aggressive quantization
    """
    # Load the model
    if isinstance(model_path, str):
        checkpoint = torch.load(model_path, map_location='cpu')
        # Assume model is stored in checkpoint['state_dict']
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    else:
        model = model_path
    
    # Quantize
    model = quantize_model_inference(model, aggressive=aggressive)
    
    # Set to eval mode
    model.eval()
    
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False
    
    # Optimize memory usage
    if torch.cuda.is_available():
        model = model.cuda()
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Save optimized model
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.__dict__,
            'quantization_info': {
                'dtype': 'float16',
                'aggressive': aggressive,
                'optimization_date': str(torch.datetime.now())  # Optional: replace if datetime not supported
            }
        }, save_path)
        logging.info(f"Optimized model saved to {save_path}")
    
    return model


# Additional utilities for monitoring
def check_model_memory_usage(model: nn.Module) -> Dict[str, Any]:
    """Check memory usage of the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters by dtype
    fp32_params = sum(p.numel() for p in model.parameters() if p.dtype == torch.float32)
    fp16_params = sum(p.numel() for p in model.parameters() if p.dtype == torch.float16)
    
    memory_fp32 = fp32_params * 4 / (1024**3)  # GB
    memory_fp16 = fp16_params * 2 / (1024**3)  # GB
    total_memory = memory_fp32 + memory_fp16
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'fp32_parameters': fp32_params,
        'fp16_parameters': fp16_params,
        'memory_usage_gb': {
            'fp32': memory_fp32,
            'fp16': memory_fp16,
            'total': total_memory
        },
        'memory_saved_gb': fp32_params * 2 / (1024**3)  # Estimated memory saved by fp16
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Assuming we have a model instance
    # model = LatentDiffusion(...)  # Your model here
    
    # Quantize model
    # quantized_model = quantize_model_inference(model, aggressive=False)
    
    # Check memory usage
    # memory_info = check_model_memory_usage(quantized_model)
    # print("Memory usage info:", memory_info)
    
    print("Quantization code is ready for use!")
