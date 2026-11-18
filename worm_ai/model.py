"""WormBrainNet: Fast CPU Dual-Stream Spatial Dilated ResNet + 1-Step Transition Actor-Critic."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

from .physics import ACTION_FEATURES, GRID_CHANNELS, H, W


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
        )
        self.se = SqueezeExcite(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.se(self.block(x))


class WormBrainNet(nn.Module):
    """
    Shared Policy-Value Network combining:
    1) Global 12x12x22 Multi-Scale Dilated ResNet (dilation 1, 2, 4 -> full-board receptive field)
    2) Head-centric + Portal-centric + Global spatial token readout
    3) 4 x 26 1-step simulator transition lookahead features with lossless safe-action masking
    """

    def __init__(self, channels: int = 24):
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
        )
        self.res1 = DilatedResBlock(channels, dilation=1)
        self.res2 = DilatedResBlock(channels, dilation=2)
        self.res3 = DilatedResBlock(channels, dilation=4)

        self.spatial_proj = nn.Sequential(
            nn.Linear(channels * 4, 80),
            nn.LayerNorm(80),
            nn.SiLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(80 + ACTION_FEATURES, 80),
            nn.SiLU(),
            nn.Linear(80, 40),
            nn.SiLU(),
            nn.Linear(40, 1),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(80 + 4 * ACTION_FEATURES, 96),
            nn.SiLU(),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

    def encode_spatial(self, grid: torch.Tensor) -> torch.Tensor:
        feat = self.stem(grid)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.res3(feat)

        # Keypoint token pooling at worm head (ch 5) and portal (ch 4)
        head_mask = grid[:, 5:6]
        portal_mask = grid[:, 4:5]
        head_tok = (feat * head_mask).sum(dim=(2, 3))
        portal_tok = (feat * portal_mask).sum(dim=(2, 3)) / portal_mask.sum(dim=(2, 3)).clamp_min(1.0)
        avg_tok = feat.mean(dim=(2, 3))
        max_tok = feat.amax(dim=(2, 3))

        merged = torch.cat((head_tok, portal_tok, avg_tok, max_tok), dim=-1)
        return self.spatial_proj(merged)

    def forward(
        self,
        grid: torch.Tensor,
        action_feats: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.encode_spatial(grid)  # (B, 96)
        ctx_exp = ctx.unsqueeze(1).expand(-1, 4, -1)
        actor_in = torch.cat((ctx_exp, action_feats), dim=-1)
        raw_logits = self.actor_head(actor_in).squeeze(-1)
        # Absorbing win prior when an immediate step reaches the portal safely (idx 12 == 1.0)
        logits = raw_logits + 8.0 * action_feats[:, :, 12]
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e6)

        critic_in = torch.cat((ctx, action_feats.flatten(1)), dim=-1)
        value = self.critic_head(critic_in).squeeze(-1)
        return logits, value

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_brain_checkpoint(
    path: Path | str,
    model: WormBrainNet,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {
        "architecture": "WormBrainNet_v1",
        "channels": model.channels,
        "model": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, tmp)
    tmp.replace(path)


def load_brain_checkpoint(
    path: Path | str,
    device: str | torch.device = "cpu",
) -> tuple[WormBrainNet, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    channels = int(payload.get("channels", 24))
    model = WormBrainNet(channels=channels).to(device)
    model.load_state_dict(payload["model"])
    return model, payload.get("metadata", {})
