import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std"]


def _flatten_samples(samples: list) -> list[Sample]:
    """Flatten nested sample lists, taking the last sample from each sublist."""
    flat = []
    for item in samples:
        if isinstance(item, list):
            flat.append(item[-1] if item else None)
        else:
            flat.append(item)
    return [s for s in flat if s is not None]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    flat_samples = _flatten_samples(samples)
    rewards = [sample.get_reward_value(args) for sample in flat_samples]
    keep = torch.tensor(rewards, dtype=torch.float).std() > 0.0
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
