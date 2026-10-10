#!/usr/bin/env python3
"""Behavioral counterexamples for the production quiet extension code."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"source/rl_training"))
from m20_quiet.extended import ExtendedQuietSettings, ExtendedQuietTracker, surface_normals, extended_cost, action_group_rate


def main():
    torch.set_num_threads(1)
    settings = ExtendedQuietSettings()
    wheels = torch.zeros(1, 4, 3)
    xy = torch.cartesian_prod(torch.tensor([-.1, 0., .1]), torch.tensor([-.1, 0., .1]))
    hits = torch.cat((xy, (.1*xy[:, 0])[:, None]), -1)[None]
    normals, valid = surface_normals(wheels, hits, settings)
    target = torch.tensor([-.1, 0., 1.]); target /= target.norm()
    assert valid.all() and torch.allclose(normals[0, 0], target, atol=1e-6)
    missing, valid = surface_normals(wheels, torch.full_like(hits, torch.nan), settings)
    assert not valid.any() and torch.equal(missing[..., 2], torch.ones(1, 4))
    # Two flat treads must not be interpreted as a ramp through the stair face.
    stepped = hits.clone(); stepped[..., 2] = torch.where(xy[:, 0] > 0, .2, 0.)
    normals, valid = surface_normals(wheels, stepped, settings)
    assert valid.all() and torch.allclose(normals[..., :2], torch.zeros(1, 4, 2))
    tracker = ExtendedQuietTracker(1, "cpu", settings)
    zeros = torch.zeros(1, 4, 3)
    n = zeros.clone(); n[..., 2] = 1
    args = dict(event=torch.zeros(1, 4, dtype=torch.bool), contact=torch.ones(1, 4, dtype=torch.bool),
                eligible=torch.ones(1, dtype=torch.bool), force=zeros.clone(), normals=n,
                wheel_velocity=zeros.clone(), wheel_omega=zeros.clone(), radius=.09,
                wheel_acceleration=torch.zeros(1, 4), base_omega=torch.zeros(1, 3),
                base_velocity=torch.zeros(1, 3), torque=torch.zeros(1, 16),
                clearance=torch.full((1, 4), .02), map_valid=torch.ones(1, 4, dtype=torch.bool),
                swing_gate=torch.zeros(1, 4, dtype=torch.bool), dt=.005)
    args['force'][..., 2] = 50
    tracker.update(**args)
    assert all(value.sum() == 0 for value in tracker.costs.values()), "Support/birth is not impact"
    args['event'].fill_(True); args['force'][..., 2] = 300
    tracker.begin_step(); tracker.update(**args)
    peak = tracker.costs['impact_peak'].clone()
    assert peak.item() == 4 and tracker.costs['impact_impulse'].item() > 0
    assert tracker.costs['loading_rate'].item() > 0
    args['event'].zero_(); tracker.begin_step()
    for _ in range(3): tracker.update(**args)
    assert tracker.costs['impact_peak'].item() == 0, "No repeat peak charge across policy boundary"
    assert tracker.costs['impact_impulse'].item() > 0, "Window continues across policy boundary"
    for _ in range(10): tracker.update(**args)
    tracker.begin_step(); tracker.update(**args)
    assert tracker.costs['impact_impulse'].item() == 0, "Window expires"
    # Correct circular rolling cancels bottom-point velocity; center-speed would fail this.
    args['wheel_velocity'][..., 0] = 1
    args['wheel_omega'][..., 1] = 1/.09
    tracker.begin_step(); tracker.update(**args)
    assert tracker.costs['wheel_slip'].item() < 1e-10
    args['wheel_omega'].zero_(); tracker.begin_step(); tracker.update(**args)
    assert tracker.costs['wheel_slip'].item() > 0, "Sliding must be penalized"
    args['contact'].zero_(); tracker.begin_step(); tracker.update(**args)
    assert tracker.costs['wheel_slip'].item() == 0, "Airborne is not slip"
    args['swing_gate'].fill_(True); tracker.begin_step(); tracker.update(**args)
    assert tracker.costs['swing_clearance'].item() > 0
    args['map_valid'].zero_(); tracker.begin_step(); tracker.update(**args)
    assert tracker.costs['swing_clearance'].item() == 0, "Unknown map is not zero clearance"
    args['base_velocity'][..., 2] = 1; args['base_omega'][..., 0] = 1
    args['wheel_acceleration'].fill_(10); args['torque'].fill_(10)
    tracker.begin_step(); tracker.update(**args)
    for name in ('base_vertical_acceleration','base_angular_acceleration','wheel_jerk','torque_rate'):
        assert tracker.costs[name].item() > 0
    tracker.reset([0]); tracker.begin_step(); tracker.update(**args)
    for name in ('base_vertical_acceleration','base_angular_acceleration','wheel_jerk','torque_rate'):
        assert tracker.costs[name].item() == 0, "Reset discontinuity must be masked"
    for dt in (.01, .02, .04):
        env = SimpleNamespace(quiet_extended_tracker=tracker, step_dt=dt)
        assert torch.allclose(extended_cost(env, 'wheel_slip')*dt, tracker.costs['wheel_slip'])
    action = torch.zeros(1, 16); action[:, 12:] = 2
    env = SimpleNamespace(action_manager=SimpleNamespace(action=action, prev_action=torch.zeros_like(action)),
                          _quiet_wheel_action_ids=slice(12,16), _quiet_leg_action_ids=slice(0,12))
    assert action_group_rate(env,'leg').item() == 0 and action_group_rate(env,'wheel').item() == 4
    args['force'].fill_(float('nan'))
    try:
        tracker.update(**args)
        raise AssertionError("Non-finite force accepted")
    except ValueError: pass
    # Peak tail: distinguish severe impacts without changing V1 or other penalties.
    v11 = ExtendedQuietSettings(impact_peak_mode="quadratic_linear")
    x = torch.tensor([0., 1., 2., 3., 8.])
    assert torch.equal(settings.peak_cost(x), torch.tensor([0., 1., 4., 4., 4.]))
    assert torch.equal(v11.peak_cost(x), torch.tensor([0., 1., 4., 8., 28.]))
    args['force'].zero_(); args['force'][..., 2] = 600
    args['event'].fill_(True); args['contact'].fill_(True)
    args['base_velocity'].zero_(); args['base_omega'].zero_()
    args['wheel_velocity'].zero_(); args['wheel_acceleration'].zero_(); args['torque'].zero_()
    a, b = ExtendedQuietTracker(1, 'cpu', settings), ExtendedQuietTracker(1, 'cpu', v11)
    a.update(**args); b.update(**args)
    assert b.costs['impact_peak'].item() > a.costs['impact_peak'].item()
    for key in a.costs:
        if key != 'impact_peak': assert torch.equal(a.costs[key], b.costs[key]), key
    args['event'].zero_(); args['force'][..., 2] = 1000
    b.begin_step(); b.update(**args)
    assert b.costs['impact_peak'].item() > 0, 'Growing peak pays only incremental cost'
    b.begin_step(); b.update(**args)
    assert b.costs['impact_peak'].item() == 0, 'No repeat peak charge'
    # Support balance: persistent flat four-support bias is charged; swap/air/stair is not.
    v12 = ExtendedQuietSettings(weights={**settings.weights, 'load_balance': -.05})
    def balance_run(forces, gate=True, contact=True, dt=.005):
        probe = ExtendedQuietTracker(1, 'cpu', v12)
        args['force'][..., 2] = torch.tensor(forces)
        args['contact'].fill_(contact)
        args['balance_gate'] = torch.tensor([gate])
        args['dt'] = dt
        for _ in range(round(.5/dt)): probe.update(**args)
        return probe
    assert balance_run([100,100,100,100]).costs['load_balance'].item() == 0
    biased = balance_run([60,100,100,60])
    assert biased.costs['load_balance'].item() > 0
    assert biased.metrics([0])['QuietLoad/stable_diagonal_abs_mean'].item() > 0
    unstabilized = balance_run([60,100,100,60],gate=False)
    assert unstabilized.metrics([0])['QuietLoad/stable_has_samples'].item() == 0
    assert 'QuietLoad/stable_diagonal_abs_mean' not in unstabilized.metrics([0])
    assert balance_run([60,100,100,60],gate=False).costs['load_balance'].item() == 0
    assert balance_run([60,100,100,60],contact=False).costs['load_balance'].item() == 0
    assert abs(biased.costs['load_balance'].item()-balance_run([60,100,100,60],dt=.01).costs['load_balance'].item()) < .02
    args['dt'] = .005; args['contact'].fill_(True)
    args['balance_gate'].fill_(False); biased.begin_step(); biased.update(**args)
    assert biased.costs['load_balance'].item() == 0 and biased.balance_age.item() == 0
    args['balance_gate'].fill_(True); biased.update(**args)
    assert biased.costs['load_balance'].item() == 0, 'Re-entry must settle again'
    biased.reset([0])
    assert biased.balance_age.item() == 0 and biased.balance_force_mean.sum().item() == 0
    assert biased.load_force_integral.sum().item() == 0
    # Both changes must coexist in the same production tracker.
    combined = ExtendedQuietSettings(impact_peak_mode="quadratic_linear",
                                    weights={**settings.weights, 'load_balance': -.05})
    probe = ExtendedQuietTracker(1, 'cpu', combined)
    args['event'].fill_(True); args['force'][..., 2] = 1000
    probe.update(**args)
    assert probe.costs['impact_peak'].item() > a.settings.cost_cap * 4
    assert probe.costs['load_balance'].item() == 0, 'Touchdown must suspend balance'
    args['event'].zero_(); args['force'][..., 2] = torch.tensor([60,100,100,60])
    for _ in range(100): probe.update(**args)
    assert probe.costs['load_balance'].item() > 0
    assert probe.metrics([0])['QuietLoad/stable_has_samples'].item() == 1
    probe.reset([0])
    assert all(cost.sum().item() == 0 for cost in probe.costs.values())
    print('QUIET_V15_CPU_PASS: combined peak/balance, touchdown gate, reset; isolated tails and balance regressions')
    # Exercise the adapter's production POST refresh with a free-fall counterexample.
    import mujoco
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'deploy/deploy_mujoco'))
    from quiet_landing_metrics import QuietMujocoMetrics
    model = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.005"/><worldbody><body pos="0 0 2"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_step1(model,data); mujoco.mj_step2(model,data)
    old_velocity = torch.zeros(6, dtype=torch.float64).numpy()
    mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,1,old_velocity,0)
    assert old_velocity[5] == 0 and data.qvel[2] < 0, 'Counterexample must expose stale cache'
    adapter = QuietMujocoMetrics.__new__(QuietMujocoMetrics)
    adapter.model, adapter.kinematics = model, mujoco.MjData(model)
    before_acc = data.qacc.copy()
    refreshed = adapter.refresh_post_kinematics(data)
    new_velocity = old_velocity.copy()
    mujoco.mj_objectVelocity(model,refreshed,mujoco.mjtObj.mjOBJ_BODY,1,new_velocity,0)
    assert abs(new_velocity[5]-data.qvel[2]) < 1e-10
    assert (before_acc == data.qacc).all(), 'Kinematic refresh must preserve solved physics samples'
    print('QUIET_MUJOCO_POST_CACHE_PASS: stale-cache counterexample, production scratch refresh, preserved qacc')
    print('QUIET_EXTENDED_CPU_PASS: normals/step-edge, support/impact/window, rolling/slip, clearance, derivative/reset, dt, action-order, NaN')


if __name__ == '__main__': main()
