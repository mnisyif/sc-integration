import torch
import torch.nn as nn
import torchaudio

class Wav2Vec2Encoder(nn.Module):
    def __init__(self, model_name: str = 'WAV2VEC2_BASE', reduce: str = 'mean'):
        super().__init__()

        self.model_name = model_name
        self.reduce = reduce.lower() if reduce is not None else None

        print(f"Loading pretrained Wav2Vec2 model: {model_name}")
        bundle = getattr(torchaudio.pipelines, model_name)
        self.model = bundle.get_model()
        self.feature_dim = bundle._params.get("encoder_embed_dim")
        
        # Print model information to verify loading
        print(f"✓ Wav2Vec2 model loaded successfully")
        print(f"  - Model: {model_name}")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Sample rate: {bundle.sample_rate}")
        print(f"  - Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Verify the model has pretrained weights by checking if parameters are not randomly initialized
        first_param = next(self.model.parameters())
        param_mean = first_param.mean().item()
        param_std = first_param.std().item()
        print(f"  - Parameter statistics (mean={param_mean:.6f}, std={param_std:.6f})")
        
        if abs(param_mean) > 1e-6 or param_std > 0.01:
            print("  ✓ Model appears to have pretrained weights (non-zero parameters)")
        else:
            print("  ⚠ Warning: Model parameters appear to be near zero (possible initialization issue)")

    def get_model_info(self):
        """Return information about the loaded model"""
        return {
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'reduce_method': self.reduce,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_type': type(self.model).__name__
        }

    def forward(self, x: torch.Tensor):
        """
        x : (B, T) — raw waveform
        returns : (B, 1, C) or (B, T', C) if no reduction
        """
        with torch.inference_mode():
            features, _ = self.model(x)  # Wav2Vec2 returns (features, lengths)

        if self.reduce == 'mean':
            features = features.mean(dim=1, keepdim=True)
        elif self.reduce == 'max':
            features, _ = features.max(dim=1, keepdim=True)
        
        return features
    
    def output_dim(self):
        return self.feature_dim