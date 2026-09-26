"""MART + full MSE LORE on 1D signals, checked against direct reference computations."""
import pytest
import torch
import torch.nn.functional as F

from baselines import mart_loat
from baselines.adapters import BASELINES

torch.set_num_threads(2)


def module():
    return mart_loat


def reference_lore(clean, adv, y, phase):
    p, pa = clean.softmax(1), adv.softmax(1)
    groups = {True: [], False: []}
    pairs = {True: [], False: []}
    for i in range(len(y)):
        correct = bool(clean[i].argmax() == y[i])
        others = [k for k in range(clean.shape[1]) if k != int(y[i])]
        mean = sum(p[i, k] for k in others) / len(others)
        groups[correct].append(sum((p[i, k] - mean) ** 2 for k in others))
        pairs[correct].append(((pa[i] - p[i]) ** 2).sum())
    zero = clean.sum() * 0
    avg = lambda values: torch.stack(values).mean() if values else zero
    c, w = avg(groups[True]), avg(groups[False])
    if phase == "early":
        return c - w, avg(pairs[False])
    if phase == "late":
        return w - c, avg(pairs[True])
    return zero, zero


@pytest.mark.parametrize("classes", [3, 4, 10])
@pytest.mark.parametrize("group", ["mixed", "correct", "wrong"])
@pytest.mark.parametrize("phase", ["early", "middle", "late"])
def test_lore_values_and_gradients_match_reference(classes, group, phase):
    torch.manual_seed(14)
    clean = torch.randn(6, classes, dtype=torch.float64, requires_grad=True)
    adv = torch.randn(6, classes, dtype=torch.float64, requires_grad=True)
    pred = clean.detach().argmax(1)
    y = pred.clone()
    if group == "wrong":
        y = (pred + 1) % classes
    elif group == "mixed":
        y[::2] = (pred[::2] + 1) % classes
    actual = module().lore_terms(clean, adv, y, phase)
    expected = reference_lore(clean, adv, y, phase)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    a = actual[0] * 0.1 + actual[1] * 0.05 + adv.sum() * 0
    b = expected[0] * 0.1 + expected[1] * 0.05 + adv.sum() * 0
    ga = torch.autograd.grad(a, (clean, adv), retain_graph=True)
    gb = torch.autograd.grad(b, (clean, adv))
    for x, y in zip(ga, gb):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("classes", [3, 4, 10])
def test_mart_matches_published_loss(classes):
    torch.manual_seed(7)
    clean, adv = [torch.randn(8, classes, requires_grad=True) for _ in range(2)]
    y = torch.arange(8) % classes
    p, pa = clean.softmax(1), adv.softmax(1)
    top2 = pa.argsort(1)[:, -2:]
    other = torch.where(top2[:, -1] == y, top2[:, -2], top2[:, -1])
    expected = F.cross_entropy(adv, y) + F.nll_loss(torch.log(1.0001 - pa + 1e-12), other)
    kl = F.kl_div(torch.log(pa + 1e-12), p, reduction="none").sum(1)
    expected += 6 * (kl * (1.0000001 - p[torch.arange(8), y])).mean()
    actual = module().mart_loss(clean, adv, y, beta=6)
    torch.testing.assert_close(actual, expected)
    grads = torch.autograd.grad(actual, (clean, adv), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (clean, adv))
    for a, b in zip(grads, expected_grads):
        torch.testing.assert_close(a, b)


def test_mart_kl_survives_clean_softmax_underflow():
    """Logit spreads large enough for a float32 softmax to underflow must keep finite gradients."""
    clean, adv = torch.zeros(2, 10), torch.zeros(2, 10)
    clean[:, 0], clean[:, 1] = 94.76, -20.58
    adv[:, 0], adv[:, 1] = 35.58, -18.79
    clean.requires_grad_(True)
    adv.requires_grad_(True)
    y = torch.zeros(2, dtype=torch.long)
    assert (clean.detach().softmax(1) == 0).any(), "this regime must actually underflow"
    loss = module().mart_loss(clean, adv, y, beta=6)
    assert torch.isfinite(loss)
    for g in torch.autograd.grad(loss, (clean, adv)):
        assert torch.isfinite(g).all()


def test_phase_rescales_official_110_epoch_schedule():
    phase = module().loat_phase
    assert [phase(i, 110) for i in (0, 1, 98, 99, 109)] == ["early", "middle", "middle", "late", "late"]
    assert [phase(i, 648) for i in (0, 5, 6, 588, 589, 647)] == ["early", "early", "middle", "middle", "late", "late"]


def test_pgd_increases_loss_stays_in_ball_and_preserves_state():
    torch.manual_seed(5)
    net = torch.nn.Sequential(torch.nn.BatchNorm1d(1), torch.nn.Flatten(), torch.nn.Linear(12, 3)).train()
    x = torch.randn(6, 1, 12) - 2
    y = torch.arange(6) % 3
    before = {k: v.clone() for k, v in net.state_dict().items()}
    generator = torch.Generator().manual_seed(123)
    adv = module().pgd_l2(net, x, y, snr_db=20, steps=10, generator=generator)
    assert net.training
    assert all(p.grad is None for p in net.parameters())
    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, before[key])
    radii = x.flatten(1).norm(dim=1) * 0.1
    assert ((adv - x).flatten(1).norm(dim=1) <= radii + 1e-5).all()
    assert adv.min() < 0 and not adv.requires_grad
    net.eval()
    assert F.cross_entropy(net(adv), y) > F.cross_entropy(net(x), y)
    repeated = module().pgd_l2(net, x, y, snr_db=20, steps=10,
                               generator=torch.Generator().manual_seed(123))
    torch.testing.assert_close(repeated, adv)
    assert not net.training


def test_zero_signal_has_zero_radius():
    net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 3))
    x, y = torch.zeros(4, 1, 12), torch.arange(4) % 3
    torch.testing.assert_close(module().pgd_l2(net, x, y, snr_db=20, steps=2), x)


@pytest.mark.skipif(not (BASELINES / "ClassBD").is_dir(), reason="third_party/ClassBD not installed")
@pytest.mark.parametrize("classes", [3, 4, 10])
def test_real_classbd_has_finite_loss_and_gradients(classes):
    from baselines.adapters import build_classbd
    torch.manual_seed(11)
    net = build_classbd(classes, 2048).train()
    x, y = torch.randn(4, 1, 2048), torch.arange(4) % classes
    objective = module().MartLoatObjective(total_epochs=2, steps=2, seed=11)
    for epoch in (0, 1):
        net.zero_grad(set_to_none=True)
        loss = objective(net, x, y, epoch, torch.nn.CrossEntropyLoss())
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [p.grad for p in net.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


def test_unified_trainer_uses_objective_and_reports_best_checkpoint():
    import numpy as np
    from baselines.train import train_unified
    from coevolve_bearing.metrics import balanced_accuracy
    from coevolve_bearing.train import predict

    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 12)).astype("float32")
    y = np.arange(12) % 3
    cur = dict(train_X=x, train_y=y, sel_X=x, sel_y=y)
    cfg = {"data": {"dataset": "toy", "window_len": 12}, "train": {"batch_size": 6, "eval_batch_size": 6}}
    objective = module().MartLoatObjective(total_epochs=2, steps=2)
    calls, reports = [], []

    def loss_fn(model, xb, yb, epoch, criterion):
        calls.append(epoch)
        return objective(model, xb, yb, epoch, criterion)

    def report(epoch, state, best, history):
        assert best == max(history)
        reports.append((epoch, {k: v.clone() for k, v in state.items()}))

    builder = lambda n, length: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(length, n))
    state, best, history = train_unified(builder, 2, 11, cur, cfg, 3, device="cpu",
                                         loss_fn=loss_fn, epoch_callback=report)
    assert calls == [0, 0, 1, 1]
    assert [r[0] for r in reports] == [0, 1]
    for k, v in state.items():
        torch.testing.assert_close(v, reports[-1][1][k])
    net = builder(3, 12)
    net.load_state_dict(state)
    assert balanced_accuracy(y, predict(net, x, cfg)) == best


def test_failed_checkpoint_verification_does_not_mark_run_complete(tmp_path, monkeypatch):
    import numpy as np
    from baselines import run_mart_loat as runner

    x, y = np.ones((6, 12), dtype="float32"), np.arange(6) % 3
    cur = dict(train_X=x, train_y=y, sel_X=x, sel_y=y)
    cfg = {"data": {"window_len": 12}, "train": {"eval_batch_size": 6},
           "mart_loat": {"steps": 2, "attack_snr_db": 20.0, "beta": 6.0, "theta": 0.1, "gamma": 0.05}}
    monkeypatch.setattr(runner, "load_everything", lambda *args: (cfg, None, cur, 3))
    builder = lambda n, length: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(length, n))
    monkeypatch.setattr(runner, "build_classbd", builder)

    def fake_train(*args, **kwargs):
        state = builder(3, 12).state_dict()
        kwargs["epoch_callback"](0, state, 2.0, [2.0])
        return state, 2.0, [2.0]                  # an impossible recorded score

    monkeypatch.setattr(runner, "train_unified", fake_train)
    args = type("Args", (), {"out": str(tmp_path), "epochs": 1, "device": "cpu", "evaluate": False})
    with pytest.raises(RuntimeError, match="checkpoint reload"):
        runner.run_one(args, "cwru", 11)
    assert not (tmp_path / "cwru_s11" / "result.json").exists()
