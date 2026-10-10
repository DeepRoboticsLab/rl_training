#!/usr/bin/env python3
"""CPU behavioral checks of the actual shared touchdown production code."""
from pathlib import Path
import sys
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from m20_quiet.signals import QuietLandingSettings, TouchdownTracker, map_clearance


def main():
    torch.set_num_threads(1)
    settings = QuietLandingSettings(reset_grace_time=0)
    tracker = TouchdownTracker(2, "cpu", settings)
    velocity = torch.zeros((2, 4, 3))
    force = torch.zeros_like(velocity)
    force[..., 2] = 50
    tracker.update(velocity, force, .005)
    assert tracker.count.sum() == 0, "Birth contact must not create events"
    velocity[..., 0] = 2
    for _ in range(4):
        tracker.update(velocity, force, .005)
    assert tracker.count.sum() == 0, "Rolling is not repeated touchdown"
    tracker.begin_step()
    zero = torch.zeros_like(force)
    for _ in range(3):
        tracker.update(velocity, zero, .005)
    velocity[..., 2] = -1
    post_stopped = torch.zeros_like(velocity)
    events, _ = tracker.update(velocity, force, .005,
                               torch.full((2, 4), torch.nan), torch.zeros((2, 4), dtype=torch.bool))
    assert events.all() and torch.equal(tracker.count, torch.ones(2, 4))
    assert torch.equal(tracker.step_cost, torch.full((2,), 16.)), "Use PRE-impact not stopped velocity"
    assert torch.equal(tracker.speed_sq_sum, torch.ones(2, 4))
    before = tracker.step_cost.clone()
    tracker.update(post_stopped, force, .005)
    assert torch.equal(before, tracker.step_cost), "No repeat payment during support"
    stats = tracker.metrics(slice(None))
    assert stats['QuietLanding/all/pre_speed_mean_m_s'] == 1
    assert stats['QuietLanding/all/specific_energy_mean_J_kg'] == .5
    assert stats['QuietLanding/all/height_window_event_coverage'] == 0
    tracker.reset([0])
    assert tracker.count[0].sum() == 0 and tracker.count[1].sum() == 4
    assert 'QuietLanding/all/pre_speed_mean_m_s' not in tracker.metrics([0])
    tracker.begin_step()
    # A one-substep force dropout cannot re-arm a touchdown.
    tracker.update(velocity, zero, .005)
    tracker.update(velocity, force, .005)
    assert tracker.step_cost.sum() == 0
    # Real separated rebound is retained.
    for _ in range(3):
        tracker.update(velocity, zero, .005)
    tracker.update(velocity, force, .005)
    assert tracker.step_cost.sum() > 0
    # Side contacts are counted separately, not called horizontal-tread landings.
    tracker.begin_step()
    for _ in range(3):
        tracker.update(velocity, zero, .005)
    side = zero.clone()
    side[..., 0] = 50
    tracker.update(velocity, side, .005)
    assert tracker.step_cost.sum() == 0 and tracker.non_tread_edges.sum() > 0
    wheels = torch.zeros((1, 4, 3))
    wheels[..., 2] = settings.wheel_radius + .02
    hits = torch.tensor([[[0., 0., 0.], [float('nan'), 0., 0.]]])
    clearance, valid = map_clearance(wheels, hits, settings)
    assert valid.all() and torch.allclose(clearance, torch.full((1, 4), .02))
    _, valid = map_clearance(wheels, torch.full_like(hits, torch.nan), settings)
    assert not valid.any(), "Missing map must not create a perfect touchdown estimate"
    try:
        tracker.update(torch.full_like(velocity, torch.nan), force, .005)
        raise AssertionError("Non-finite sample must fail closed")
    except ValueError:
        pass
    # Discrete event contribution stays constant after RewardManager's dt factor.
    from m20_quiet.signals import touchdown_cost
    tracker.step_cost.fill_(4)
    for policy_dt in (.01, .02, .04):
        env = SimpleNamespace(quiet_tracker=tracker, step_dt=policy_dt)
        assert torch.allclose(touchdown_cost(env) * policy_dt, torch.full((2,), 4.))
    # Exercise the real sparse logger without importing the Isaac-only package.
    spec = importlib.util.spec_from_file_location("quiet_logger_check", ROOT / "source/rl_training/rl_training/quiet_landing_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    logger = module.QuietLandingLogger.__new__(module.QuietLandingLogger)
    scalars = {}
    logger.writer = SimpleNamespace(add_scalar=lambda key, value, iteration: scalars.update({key: (float(value), iteration)}))
    logger.device = 'cpu'
    logger.ep_extras = [{}, {'QuietLanding/all/touchdown_count': 2},
                        {'QuietLanding/all/pre_speed_mean_m_s': 1.3},
                        {'QuietLanding/all/pre_speed_mean_m_s': .7},
                        {'QuietLanding/all/pre_speed_mean_m_s': float('nan')}]
    with patch.object(module.Logger, 'log', return_value=None):
        logger.log(it=42)
    assert scalars['QuietLanding/all/pre_speed_mean_m_s'] == (1., 42)
    assert len(scalars) == 2 and not logger.ep_extras
    print("QUIET_LANDING_CPU_PASS: birth, rolling, pre-impact, repeat/reset, rebound, side-contact, map, dt")
    print("QUIET_SPARSE_LOGGER_PASS: empty-first, conditional keys, integer counts, NaN, iteration")


if __name__ == '__main__':
    main()
