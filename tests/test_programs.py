"""The seed programs satisfy the contracts their evaluators enforce."""
import numpy as np
import torch

import evaluator_gen
import evaluator_solver
from challenger import SEED_GEN, SEED_SOLVER


def test_challenger_seed_meets_the_contract():
    gen = evaluator_gen.load_generator_from_source(SEED_GEN.read_text())
    evaluator_gen._check_contract(gen, 12000, 2048)
    noise = gen(np.random.default_rng(0), 12000.0, 2048)
    assert noise.shape == (2048,) and np.isfinite(noise).all()


def test_solver_seed_is_the_classbd_anchor():
    torch.manual_seed(0)
    builder = evaluator_solver.load_builder_from_source(SEED_SOLVER.read_text())
    model = evaluator_solver._build_and_check(builder, 3, 2048)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 4_268_568
    model.train()
    out = model(torch.randn(4, 1, 2048))
    assert model.aux_k is not None and model.aux_g is not None
    torch.testing.assert_close(out.sum(1), torch.ones(4))


def test_seed_eval_depends_on_the_source_text():
    a = evaluator_solver.seed_eval(11, "program a")
    assert a == evaluator_solver.seed_eval(11, "program a")
    assert a != evaluator_solver.seed_eval(11, "program b")
