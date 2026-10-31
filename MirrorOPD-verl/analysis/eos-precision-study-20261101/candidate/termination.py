# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""One stopping-action map shared by reference and fused vocabulary losses.

The fused representation keeps ordinary logits in their original FP32 form and
carries the rounding remainder of the merged logit in one FP64 scalar per row.
This preserves tiny stopping-class differences without dense FP64 workspaces.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TerminationActionMap:
    keep: torch.Tensor
    stop: torch.Tensor
    vocab: int

    @classmethod
    def create(cls, vocab, ids, device):
        ids = tuple(sorted(ids))
        if len(ids) < 2 or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < vocab for i in ids):
            raise ValueError("termination equivalence requires distinct supported token IDs")
        boundaries = (-1, *ids, vocab)
        keep = torch.cat(
            [torch.arange(a + 1, b, device=device) for a, b in zip(boundaries, boundaries[1:], strict=False)]
        )
        return cls(keep, torch.tensor(ids, device=device, dtype=torch.long), vocab)

    def coarsen(self, logits, *, compensated=False):
        """Return compact logits and an optional exact rounding remainder."""
        selected = logits.index_select(1, self.stop).double()
        maximum, pivot = selected.max(-1, keepdim=True)
        shifted = (selected - maximum).scatter(-1, pivot, -torch.inf)
        tail = torch.log1p(shifted.exp().sum(-1, keepdim=True))
        if compensated:
            merged = (maximum + tail).to(logits.dtype)
            # Do not first add the tiny tail to a large FP64 maximum.
            remainder = ((maximum - merged.double()) + tail).squeeze(-1).contiguous()
            return torch.cat((logits.index_select(1, self.keep), merged), -1), remainder
        return torch.cat((logits.index_select(1, self.keep).double(), maximum + tail), -1), None

    def expand_gradient(self, gradient, logits):
        """Apply the exact logsumexp chain rule to the stopping class."""
        result = gradient.new_empty((len(logits), self.vocab))
        result.index_copy_(1, self.keep, gradient[:, :-1])
        conditional = logits.index_select(1, self.stop).double().softmax(-1).to(gradient.dtype)
        result.index_copy_(1, self.stop, gradient[:, -1:] * conditional)
        pivot = logits.argmax(-1, keepdim=True)
        other = result.scatter(-1, pivot, 0.0)
        return other.scatter(-1, pivot, -other.sum(-1, keepdim=True))
