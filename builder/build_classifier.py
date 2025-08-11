import torch
import torch.nn as nn
import yaml
import os

from net.cloud.classifier import SwinClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_classifier(config_path: str = "build/config.yaml", device=device) -> nn.Module:
    """
    Build a classifier from config.yaml and load weights if they exist.
    
    Args:
        config_path: Path to the config.yaml file
        device: Device to load the model on
    
    Returns:
        The classifier model with loaded weights (if available)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None or 'classifier' not in config or config['classifier'] is None:
        raise ValueError("The 'classifier' section is missing or invalid in the config file.")

    classifier_cfg = config['classifier']
    kwargs = classifier_cfg.get('kwargs', {}).copy()
    weights_path = classifier_cfg.get('weights', None)

    print(f"Building classifier with parameters: {kwargs}")
    
    # Build the classifier
    model = SwinClassifier(**kwargs)

    # Load weights if specified and exists
    if weights_path:
        model = load_classifier_weights(model, device, weights_path)

    return model.to(device)

def load_classifier_weights(model: nn.Module, device, weights_path: str, strict: bool = True) -> nn.Module:
    """
    Load pretrained weights into a classifier model.
    
    Args:
        model: The classifier model to load weights into
        device: Device to load weights on
        weights_path: Path to the weights file
        strict: Whether to strictly enforce that the keys in state_dict match the keys in model
    
    Returns:
        The model with loaded weights
    """
    if weights_path and os.path.exists(weights_path):
        print(f"Loading classifier weights from: {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=strict)
            print("✓ Classifier weights loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load classifier weights: {e}")
    elif weights_path:
        print(f"⚠ Warning: Weights file not found at {weights_path}")
    
    return model