from net.modulator.netaware import NetAwareMod
from net.modulator.policies import LogBandPolicy

def build_modulator(mod_cfg: dict):
    policy_cfg = mod_cfg["policy"]
    policy_type = policy_cfg.pop("type", "logband").lower()

    if policy_type == "logband":
        policy = LogBandPolicy(**policy_cfg)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    if mod_cfg["type"] == "netaware":
        return NetAwareMod(policy, auto_unit=mod_cfg.get("auto_unit", True))
    else:
        raise ValueError(f"Unknown modulator type: {mod_cfg['type']}")
