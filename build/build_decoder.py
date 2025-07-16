import torch
import torch.nn as nn
import os
import yaml

from net.cloud import swin

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_decoder(decoder_name: str, config_path: str = "build/config.yaml", device = device) -> nn.Module:
    """
    Build an encoder from config.yaml and load weights if they exist.
    
    Args:
        encoder_name: Name of the encoder to build. If None, uses the one specified in config.yaml
        config_path: Path to the config.yaml file
        device: Device to load the model on
    
    Returns:
        The encoder model with loaded weights (if available)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None or 'decoders' not in config or config['decoders'] is None:
        raise ValueError("The 'decoders' section is missing or invalid in the config file.")

    decoder_cfg = config['decoders']

    # Use provided decoder name or default from config
    name = decoder_name if decoder_name is not None else decoder_cfg['name']
    kwargs = decoder_cfg.get('kwargs', {}).copy()  # Create a copy to avoid modifying original
    weights_path = decoder_cfg.get('weights', None)

    # Handle norm_layer string conversion
    if 'norm_layer' in kwargs and isinstance(kwargs['norm_layer'], str):
        if kwargs['norm_layer'] in ['LayerNorm', 'nn.LayerNorm']:
            kwargs['norm_layer'] = nn.LayerNorm
        # Add other norm layer mappings if needed

    print(f"Building encoder: {name} with parameters: {kwargs}")
    
    # Build the encoder
    decoder_map = {
        'swin': lambda: swin.create_decoder(**kwargs),
        # 'wav2vec2': lambda: wav2vec2.Wav2Vec2Encoder(**kwargs),
        # Add other encoders here as needed
    }

    if name in decoder_map:
        model = decoder_map[name]()
    else:
        raise ValueError(f"Unknown decoder type: {name}")

    # Load weights if specified and exists
    if weights_path:
        model = load_decoder_weights(model, device, weights_path)

    return model

def load_decoder_weights(model: nn.Module, device, weights_path: str, strict: bool = False) -> nn.Module:
    """
    Load pretrained weights into a decoder model.
    
    Args:
        model: The decoder model to load weights into
        device: Device to load weights on
        weights_path: Path to the weights file
        strict: Whether to strictly enforce that the keys in state_dict match the keys in model
    
    Returns:
        The model with loaded weights
    """
    if weights_path and os.path.exists(weights_path):
        print(f"Loading decoder weights from: {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=strict)
            print("✓ Decoder weights loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load decoder weights: {e}")
    elif weights_path:
        print(f"⚠ Warning: Weights file not found at {weights_path}")
    
    return model
