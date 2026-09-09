"""Small, testable direct-agent objectives; existing CoMLRL trainers are untouched."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def token_logps(model, prompt: torch.Tensor, completion: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Every completion token, including the final/EOS token, has exactly one target."""
    if prompt.numel() < 1 or completion.numel() < 1:
        raise ValueError("prompt and completion must be nonempty")
    device = next(model.parameters()).device
    prompt, completion = prompt.to(device), completion.to(device)
    inputs = torch.cat([prompt, completion[:-1]]).unsqueeze(0)
    # Qwen3 supports selective logits; avoid materializing logits for the prompt.
    out = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                use_cache=False, logits_to_keep=int(completion.numel()))
    logits = out.logits[0, -completion.numel():].float()
    if logits.shape[0] != completion.numel():
        raise ValueError("Incomplete completion-token likelihood")
    logps = -F.cross_entropy(logits, completion, reduction="none")
    if mask is None:
        return logps
    mask = mask.to(device=logps.device, dtype=torch.bool)
    if mask.shape != logps.shape or not mask.any():
        raise ValueError("A matching, nonempty value-token loss mask is required")
    return logps[mask]


def normalized(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def grpo_loss(new: torch.Tensor, old: torch.Tensor, advantage: float,
              clip: float = 0.2) -> torch.Tensor:
    ratio = torch.exp(new - old.to(new).detach())
    adv = torch.as_tensor(advantage, device=new.device, dtype=new.dtype)
    return -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()


def dpo_loss(delta: torch.Tensor, reference_delta: float, beta: float) -> torch.Tensor:
    return -F.logsigmoid(beta * (delta - reference_delta))


def dpo_delta_gradient(delta: float, reference_delta: float, beta: float) -> float:
    # Exact scalar derivative enables winner/loser backward one graph at a time.
    return float(-beta * torch.sigmoid(torch.tensor(-beta * (delta - reference_delta), dtype=torch.float64)))


class ScalarModel(torch.nn.Module):
    """Transformer hidden state + scalar head, used for V(prompt) or reward(full response)."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        param = next(backbone.parameters())
        self.head = torch.nn.Linear(backbone.config.hidden_size, 1, device=param.device, dtype=param.dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        ids = ids.to(device).unsqueeze(0)
        # Calling the base transformer avoids the unused language-model vocabulary head.
        hidden = self.backbone.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     use_cache=False).last_hidden_state
        return self.head(hidden[0, -1]).float().squeeze(-1)
