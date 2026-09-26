"""Solver seed: the official ClassBD network (BDWDCNN) as an evolvable program.

A line-by-line port of the published ClassBD model: a quadratic-convolution time-domain filter,
a learned frequency-domain linear filter with envelope-spectrum physics terms, and a WDCNN
classifier.  Published behaviour is kept as released:
  - forward returns F.softmax(out) (cross-entropy is applied on top, as published);
  - the kurtosis and lp/lq terms are wrapped in nn.Parameter, i.e. detached from the graph, and
    stored as self.aux_k / self.aux_g on every forward for the uncertainty-weighted loss;
  - the Hilbert envelope mutates its frequency-domain input in place before ifft2;
  - quadratic-convolution weights use the published normal(0, sqrt(0.25 / fan_in) * 8) init.

Contract (evolve/evaluator_solver.py): build_solver(n_classes, length) -> nn.Module mapping
(B, 1, length) -> (B, n_classes) class probabilities; parameters under the cap; torch, nn, F and
solver_lib only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import solver_lib


# EVOLVE-BLOCK-START
def build_solver(n_classes: int, length: int) -> nn.Module:
    """Official ClassBD (BDWDCNN) port: quadratic-conv time filter + learned frequency-domain
    Linear filter (with envelope-spectrum physics terms) + WDCNN classifier. Evolve the
    composition/structure; keep populating self.aux_k / self.aux_g if a denoising front-end
    remains (the trainer adds them to the loss when present)."""
    FS = 12000.0  # sampling rate of every dataset after resampling

    def _official_cqo(in_ch, out_ch, k):
        m = solver_lib.ConvQuadraticOperation(in_ch, out_ch, k)
        nn.init.normal_(m.weight_r, mean=0.0, std=(0.25 / (in_ch * k)) ** 0.5 * 8)
        return m

    def _kurtosis(y, half=32):
        y_1 = torch.squeeze(y)
        y_1 = y_1[:, half:-half]
        y_2 = y_1 - torch.mean(y_1)
        num = len(y_2)
        y_num = torch.sum(torch.pow(y_2, 4), dim=-1) / num
        std = torch.sqrt(torch.sum(torch.pow(y_2, 2), dim=-1) / num)
        return nn.Parameter((y_num / torch.pow(std, 4)).mean())

    def _env_spectrum(x):
        n = x.shape[-1]
        if n % 2 == 0:
            x[..., 1:n // 2] *= 2
            x[..., n // 2 + 1:] = 0
        else:
            x[..., 1:(n + 1) // 2] *= 2
            x[..., (n + 1) // 2:] = 0
        analytic = torch.fft.ifft(x)
        envelope = analytic.abs()
        en = envelope - torch.mean(envelope)
        return torch.abs(torch.fft.fft(en)) * 2 / len(en)

    def _g_lplq(es_raw_abs, p=2, q=4):
        p = torch.tensor(float(p))
        q = torch.tensor(float(q))
        obj = torch.sign(torch.log(q / p)) * (
            torch.norm(es_raw_abs, p, dim=-1) / torch.norm(es_raw_abs, q, dim=-1)) ** p
        return nn.Parameter(obj.mean())

    class OfficialClassBDSeed(nn.Module):
        def __init__(self):
            super().__init__()
            # CLASSBD time-domain quadratic filter (AvgPool1d(1,1)/MaxPool1d(1,1) no-ops dropped)
            self.qtfilter = nn.Sequential(
                _official_cqo(1, 16, 63),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                _official_cqo(16, 1, 63),
                nn.BatchNorm1d(1),
                nn.ReLU(),
                nn.Sigmoid(),
            )
            # CLASSBD frequency-domain learned filter (the parameter giant: length x length)
            self.filter1 = nn.Linear(length, length)
            # WDCNN backbone
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 16, 64, 8, 28), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2, 2),
                nn.Conv1d(16, 32, 3, 1, 1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2, 2),
                nn.Conv1d(32, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
                nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
                nn.Conv1d(64, 64, 3, 1, 1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
                nn.Conv1d(64, 64, 3, 1, 0), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, 2),
            )
            self.fc1 = nn.Linear(64 * 3, 100)
            self.relu1 = nn.ReLU()
            self.dp = nn.Dropout(0.5)
            self.fc2 = nn.Linear(100, n_classes)
            self.aux_k = None
            self.aux_g = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # time filter + kurtosis term
            a1 = self.qtfilter(x)
            k = _kurtosis(a1)
            # frequency filter + envelope-spectrum sparsity term (in-place hilbert, as published)
            f1 = torch.fft.fft2(a1 - torch.mean(a1, dim=-1).unsqueeze(1))
            f1 = abs(f1)
            es = self.filter1(f1)
            es_raw_abs = _env_spectrum(es)
            a2 = abs(torch.fft.ifft2(es))
            g = _g_lplq(es_raw_abs)
            # detach before stashing: k/g are nn.Parameter (published quirk => already cut off
            # from the model graph); assigning a raw nn.Parameter to a module attribute would
            # REGISTER it into state_dict and break weight reload on a fresh instance
            self.aux_k = (-k).detach()
            self.aux_g = g.detach()
            # backbone
            out = self.cnn(a2)
            out = self.fc1(out.view(x.size(0), -1))
            out = self.relu1(out)
            out = self.dp(out)
            out = self.fc2(out)
            return F.softmax(out, dim=1)

    return OfficialClassBDSeed()
# EVOLVE-BLOCK-END


def get_solver(n_classes: int, length: int) -> nn.Module:
    return build_solver(n_classes, length)
