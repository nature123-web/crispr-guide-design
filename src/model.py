"""Two-branch network for on-target guide efficiency prediction.

```
one-hot guide (4 x 20) ──► Conv1d stack ──┐
                                          ├──► MLP ──► efficiency in [0, 1]
biophysical features (14) ────────────────┘
```

The two-branch design is the point. A convolution is translation-invariant,
which is right for detecting a motif anywhere in the guide and wrong for
position-specific effects -- a G at position 20 is favourable *because it is at
position 20*, and a convolution cannot express that without wasting capacity.
The feature branch carries exactly those absolute, hand-computable quantities
(GC content, Tm, poly-T, position-20 identity), leaving the convolutions to
learn the patterns the features cannot name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import BIOPHYSICAL_FEATURE_DIM
from .sequence import GUIDE_LENGTH


class GuideEfficiencyNet(nn.Module):
    """Predicts cutting efficiency in [0, 1] from guide sequence and features."""

    def __init__(
        self,
        n_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        feature_dim: int = BIOPHYSICAL_FEATURE_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_features: bool = True,
        use_positional: bool = True,
    ) -> None:
        super().__init__()
        self.use_features = use_features
        self.use_positional = use_positional

        # Parallel kernels rather than a deep stack: guides are only 20 nt, so
        # depth exhausts the sequence quickly. Multiple widths in parallel see
        # 3-mers through 7-mers directly.
        self.convs = nn.ModuleList([
            nn.Conv1d(4, n_filters, k, padding=k // 2) for k in kernel_sizes
        ])
        conv_out = n_filters * len(kernel_sizes)
        self.conv_norm = nn.BatchNorm1d(conv_out)

        # Learned positional embedding added to the one-hot input, so the
        # convolutions can break translation invariance where it matters.
        if use_positional:
            self.position = nn.Parameter(torch.zeros(1, 4, GUIDE_LENGTH))
            nn.init.normal_(self.position, std=0.02)

        # Both mean and max pooling: max finds whether a motif is present
        # anywhere, mean captures how much of the guide supports it.
        pooled_dim = conv_out * 2
        merged_dim = pooled_dim + (feature_dim if use_features else 0)

        self.head = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, one_hot: torch.Tensor,
                features: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits; apply sigmoid for an efficiency in [0, 1]."""
        x = one_hot
        if self.use_positional:
            x = x + self.position

        activations = torch.cat([F.relu(conv(x)) for conv in self.convs], dim=1)
        activations = self.conv_norm(activations)
        pooled = torch.cat([
            activations.mean(dim=-1), activations.amax(dim=-1)
        ], dim=-1)

        if self.use_features:
            if features is None:
                raise ValueError(
                    "model was built with use_features=True but no features "
                    "were passed"
                )
            pooled = torch.cat([pooled, features], dim=-1)

        return self.head(pooled).squeeze(-1)

    def predict_efficiency(self, one_hot: torch.Tensor,
                           features: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self(one_hot, features))


def build_model(cfg: dict) -> GuideEfficiencyNet:
    m = cfg["model"]
    return GuideEfficiencyNet(
        n_filters=m["n_filters"], kernel_sizes=tuple(m["kernel_sizes"]),
        hidden_dim=m["hidden_dim"], dropout=m["dropout"],
        use_features=m["use_features"], use_positional=m["use_positional"],
    )
