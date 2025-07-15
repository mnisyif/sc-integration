# net/bw_log_policy.py
import math
from typing import Sequence
from pydantic import BaseModel, Field

class LogBandPolicy(BaseModel):
    """
    Pure log-bandwidth → k  (latency ignored)

    Parameters
    ----------
    B_min   : lower anchor in Mb/s
    B_max   : upper anchor in Mb/s
    gamma   : >1 steepens the curve (γ=1 ⇒ equal log bins)
    levels  : the six legal k values
    """
    B_min : float = 10.0
    B_max : float = 1_000.0
    gamma : float = 2.0
    levels: Sequence[int] = (32, 64, 96, 128, 160, 192)
    
    class Config:
        frozen = True

    def __call__(self, B: float, _lat_unused: float | None = None) -> int:
        B = max(min(B, self.B_max), self.B_min)

        log_norm = (
            (math.log10(B) - math.log10(self.B_min)) /
            (math.log10(self.B_max) - math.log10(self.B_min))
        )

        score = log_norm ** self.gamma                # γ > 1  → steeper
        idx = int(score * len(self.levels))           # 0 … 5
        
        if idx == len(self.levels):                   # catch score==1
            idx -= 1
            
        return self.levels[idx]
