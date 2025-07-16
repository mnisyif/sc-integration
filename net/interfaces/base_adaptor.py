import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple

class BaseAdapter(ABC, nn.Module):
    """Base class for all adapters"""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, features):
        """Transform features from encoder to model format"""
        pass
    
    @abstractmethod
    def get_output_shape(self, input_shape) -> Tuple:
        """Return expected output shape given input shape"""
        pass