"""Independent ManagerBasedRLEnv with explicit pre/post physics sampling.

The step lifecycle follows installed IsaacLab ManagerBasedRLEnv. Sampling is
inside its substep loop so rewards never confuse post-impact with pre-impact
velocity. No global hooks or changes to the teacher environments are used.
"""
import torch

from isaaclab.envs import ManagerBasedRLEnv
from m20_quiet.signals import LEGS, TouchdownTracker, map_clearance
from m20_quiet.extended import ExtendedQuietTracker, surface_normals


class M20QuietLandingEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.quiet_tracker = TouchdownTracker(self.num_envs, self.device, cfg.quiet_landing)
        self.quiet_extended_tracker = ExtendedQuietTracker(self.num_envs, self.device, cfg.quiet_extended)
        robot = self.scene["robot"]
        sensor = self.scene.sensors["contact_forces"]
        self._quiet_body_ids, names = robot.find_bodies([f"{leg}_wheel" for leg in LEGS], preserve_order=True)
        self._quiet_contact_ids, contact_names = sensor.find_bodies([f"{leg}_wheel" for leg in LEGS], preserve_order=True)
        if names != [f"{leg}_wheel" for leg in LEGS] or contact_names != names:
            raise RuntimeError("Quiet landing wheel order does not match FL/FR/HL/HR")
        self._quiet_wheel_joint_ids, _ = robot.find_joints([f"{leg}_wheel_joint" for leg in LEGS], preserve_order=True)
        self._quiet_leg_joint_ids, _ = robot.find_joints("^(?!.*_wheel_joint$).*")
        # Action order is 12 leg positions then 4 wheel velocities, independent
        # of the articulation joint order.
        if list(self.action_manager.action_term_dim) != [12, 4]:
            raise RuntimeError("Quiet grouped action reward requires leg12 / wheel4 action order")
        self._quiet_leg_action_ids = slice(0, 12)
        self._quiet_wheel_action_ids = slice(12, 16)
        if self.observation_manager.group_obs_dim["policy"] != (244,) or self.action_manager.total_action_dim != 16:
            raise RuntimeError("Quiet V1 requires the baseline 244D actor / 16D action contract")

    def step(self, action):
        self.extras.pop("log", None)
        self.quiet_tracker.begin_step()
        self.quiet_extended_tracker.begin_step()
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        robot = self.scene["robot"]
        sensor = self.scene.sensors["contact_forces"]
        # Original scanner only, at its original update frequency; no privileged rays.
        hits = self.scene.sensors["height_scanner"].data.ray_hits_w
        finite_hits = torch.isfinite(hits).all(-1)
        high = torch.where(finite_hits, hits[..., 2], -torch.inf).amax(-1)
        low = torch.where(finite_hits, hits[..., 2], torch.inf).amin(-1)
        stair = (high-low > self.cfg.quiet_extended.stair_height_threshold) & finite_hits.any(-1)
        command = self.command_manager.get_command("base_velocity")
        translating = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.1
        # Diagnostic/reward gate only; never appended to policy observations.
        flat = finite_hits.all(-1) & (high-low <= self.cfg.quiet_extended.load_balance_flat_range)
        start_xy = robot.data.root_pos_w[:, :2].clone()
        for _ in range(self.cfg.decimation):
            pre_velocity = robot.data.body_lin_vel_w[:, self._quiet_body_ids].clone()
            wheels = robot.data.body_pos_w[:, self._quiet_body_ids]
            clearance, valid = map_clearance(wheels, hits, self.cfg.quiet_landing)
            normals, normal_valid = surface_normals(wheels, hits, self.cfg.quiet_extended)
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            post_force = sensor.data.net_forces_w[:, self._quiet_contact_ids]
            event, _ = self.quiet_tracker.update(pre_velocity, post_force, self.physics_dt, clearance, valid, normals)
            swing_gate = (stair & translating)[:, None] & ((pre_velocity*normals).sum(-1) > 0.05)
            self.quiet_extended_tracker.update(
                event=event, contact=self.quiet_tracker.contact,
                eligible=self.quiet_tracker.age >= self.cfg.quiet_landing.reset_grace_time,
                force=post_force, normals=normals,
                wheel_velocity=robot.data.body_lin_vel_w[:, self._quiet_body_ids],
                wheel_omega=robot.data.body_ang_vel_w[:, self._quiet_body_ids],
                radius=self.cfg.quiet_landing.wheel_radius,
                wheel_acceleration=robot.data.joint_acc[:, self._quiet_wheel_joint_ids],
                base_omega=robot.data.root_ang_vel_w, base_velocity=robot.data.root_lin_vel_w,
                torque=robot.data.applied_torque, clearance=clearance, map_valid=valid,
                swing_gate=swing_gate, dt=self.physics_dt,
                balance_gate=flat & normal_valid.all(-1) & (command[:, 2].abs() < 0.1)
                & (robot.data.projected_gravity_b[:, 2] < -0.98))
            self.quiet_normal_valid_fraction = normal_valid.float().mean()
        self.quiet_tracker.distance += torch.linalg.vector_norm(robot.data.root_pos_w[:, :2] - start_xy, dim=-1)
        # Preserve this step's events for external probes even when an env resets
        # below; episode tracker reset must not erase the returned step evidence.
        self.quiet_step_events = self.quiet_tracker.step_events.clone()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_ids)
            self._reset_idx(reset_ids)
            rerenders = getattr(self.cfg, "num_rerenders_on_reset", int(getattr(self.cfg, "rerender_on_reset", False)))
            if self.sim.has_rtx_sensors():
                for _ in range(rerenders):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_ids)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids):
        tracker = getattr(self, "quiet_tracker", None)
        quiet = tracker.metrics(env_ids) if tracker is not None else {}
        extended = getattr(self, "quiet_extended_tracker", None)
        if extended is not None:
            quiet.update(extended.metrics(env_ids))
        super()._reset_idx(env_ids)
        # Keep all traversal rewards active, but do not carry their debugging
        # cards (gait_quality, angle/entry/leg pose, ascent_direction, etc.).
        old = self.extras["log"]
        kept = {key: value for key, value in old.items()
                if key.startswith("Episode_Termination/") or key in (
                    "Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw")}
        reward_total = sum(value for key, value in old.items() if key.startswith("Episode_Reward/"))
        kept["Task/episode_reward_rate"] = reward_total
        names = ("quiet_touchdown", "joint_acc_l2", "joint_acc_wheel_l2", "stair_joint_acceleration_cost",
                 "action_rate_l2", "action_smooth_l2", "quiet_action_leg_rate", "quiet_action_wheel_rate")
        names += tuple("quiet_"+name for name in self.cfg.quiet_extended.weights)
        for name in names:
            key = "Episode_Reward/" + name
            if key in old:
                kept["QuietReward/" + name] = old[key]
        kept.update(quiet)
        if hasattr(self, "quiet_normal_valid_fraction"):
            kept["QuietLanding/normal_fit_valid_fraction_last_substep"] = self.quiet_normal_valid_fraction
        self.extras["log"] = kept
        if tracker is not None:
            tracker.reset(env_ids)
        extended = getattr(self, "quiet_extended_tracker", None)
        if extended is not None:
            extended.reset(env_ids)
