"""MART with the full MSE LORE regularizer (LOAT, CVPR 2024), adapted to 1D signals.

Reimplemented from the loss equations of https://github.com/TrustAI/LOAT (commit 6ff98734,
mart/mart.py).  Signal adaptations: the PGD ball is L2 with a radius relative to each input's norm
(expressed as an SNR), cross-entropy uses the baselines' class weights, and the LORE phase
breakpoints are rescaled to the training length.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def loat_phase(epoch: int, total_epochs: int) -> str:
    """Zero-based epoch; preserve the reference's 1/110 and 100/110 breakpoints."""
    if total_epochs < 1 or not 0 <= epoch < total_epochs:
        raise ValueError("epoch must be within a positive total_epochs")
    if epoch < max(1, round(total_epochs / 110)):
        return "early"
    if epoch >= math.ceil(100 * total_epochs / 110) - 1:
        return "late"
    return "middle"


def lore_terms(logits, logits_adv, y, phase):
    """Return signed SLORE and adaptive pairing; empty groups contribute zero."""
    if phase == "middle":
        zero = (logits.sum() + logits_adv.sum()) * 0
        return zero, zero
    if phase not in ("early", "late"):
        raise ValueError(f"unknown LORE phase: {phase}")
    classes = logits.shape[1]
    p, pa = logits.softmax(1), logits_adv.softmax(1)
    correct = logits.argmax(1).eq(y)
    not_label = ~F.one_hot(y, classes).bool()
    other_mean = (1 - p.gather(1, y[:, None])) / (classes - 1)
    dispersion = ((p - other_mean).square() * not_label).sum(1)
    pairing = (pa - p).square().sum(1)

    def group_mean(values, mask):
        return (values * mask).sum() / mask.sum().clamp_min(1)

    c = group_mean(dispersion, correct)
    w = group_mean(dispersion, ~correct)
    if phase == "early":
        return c - w, group_mean(pairing, ~correct)
    return w - c, group_mean(pairing, correct)


def mart_loss(logits, logits_adv, y, beta=6.0, criterion=None):
    """MART: boosted adversarial CE + confidence-weighted KL.

    Only the CE term uses the optional class weights.  The clean distribution enters the KL
    through log_softmax: ClassBD's quadratic front end produces logit spreads large enough for a
    float32 softmax to underflow to exact zeros, where a kl_div target would give NaN gradients.
    """
    log_p = logits.log_softmax(1)
    p, pa = log_p.exp(), logits_adv.softmax(1)
    top2 = pa.topk(2, dim=1).indices
    other = torch.where(top2[:, 0].eq(y), top2[:, 1], top2[:, 0])
    ce = F.cross_entropy(logits_adv, y) if criterion is None else criterion(logits_adv, y)
    boosted = ce + F.nll_loss(torch.log(1.0001 - pa + 1e-12), other)
    kl = (p * (log_p - torch.log(pa + 1e-12))).sum(1)
    confidence = p.gather(1, y[:, None]).squeeze(1)
    return boosted + beta * (kl * (1.0000001 - confidence)).mean()


def pgd_l2(model, x, y, *, snr_db=20.0, steps=10, generator=None):
    """CE ascent inside per-input L2 balls; no [0,1] clipping of signals.

    epsilon(x) = ||x||_2 * 10^(-snr_db/20). Random direction with uniformly
    sampled radius; normalized gradient steps of 2*epsilon/steps. Attack
    forwards use eval mode and leave weights, gradients and BN statistics alone.
    """
    if steps < 1 or not math.isfinite(snr_db):
        raise ValueError("PGD needs positive steps and a finite SNR")
    x = x.detach()
    shape = (-1,) + (1,) * (x.ndim - 1)
    radius = x.flatten(1).norm(dim=1).reshape(shape) * 10 ** (-snr_db / 20)

    def norm(v):
        return v.flatten(1).norm(dim=1).reshape(shape).clamp_min(1e-12)

    def project(delta):
        return delta * (radius / norm(delta)).clamp(max=1)

    delta = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    scale = torch.rand(radius.shape, dtype=x.dtype, device=x.device, generator=generator)
    delta = delta / norm(delta) * radius * scale
    modes = [(part, part.training) for part in model.modules()]
    model.eval()
    try:
        with torch.enable_grad():
            for _ in range(steps):
                adv = (x + delta).detach().requires_grad_(True)
                loss = F.cross_entropy(model(adv), y)
                grad, = torch.autograd.grad(loss, adv)
                delta = project(adv.detach() - x + (2 * radius / steps) * grad / norm(grad))
    finally:
        for part, training in modes:
            part.training = training
    return (x + delta).detach()


@dataclass
class MartLoatObjective:
    total_epochs: int
    steps: int = 10
    snr_db: float = 20.0
    beta: float = 6.0
    theta: float = 0.1
    gamma: float = 0.05
    seed: int = 11

    def __post_init__(self):
        if self.total_epochs < 1 or self.steps < 1:
            raise ValueError("epochs and attack steps must be positive")
        if not all(math.isfinite(v) for v in (self.snr_db, self.beta, self.theta, self.gamma)):
            raise ValueError("SNR and loss coefficients must be finite")
        if min(self.beta, self.theta, self.gamma) < 0:
            raise ValueError("loss coefficients must be nonnegative")
        self._generator = None

    def __call__(self, model, x, y, epoch, criterion):
        if self._generator is None:
            self._generator = torch.Generator(device=x.device).manual_seed(self.seed + 700001)
        adv = pgd_l2(model, x, y, snr_db=self.snr_db, steps=self.steps, generator=self._generator)
        logits, logits_adv = model(x), model(adv)
        loss = mart_loss(logits, logits_adv, y, self.beta, criterion)
        standard, pairing = lore_terms(logits, logits_adv, y, loat_phase(epoch, self.total_epochs))
        return loss + self.theta * standard + self.gamma * pairing
