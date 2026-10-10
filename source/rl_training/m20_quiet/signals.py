"""Substep touchdown accounting shared by Isaac Lab and MuJoCo.

V1 measures incoming wheel-centre velocity along the estimated support normal
(vertical by default). It does not measure sound or effective contact mass.
"""
from dataclasses import dataclass
import math

import torch


LEGS = ("fl", "fr", "hl", "hr")


@dataclass
class QuietLandingSettings:
    # Initial conservative settings, to be calibrated against the pinned baseline.
    velocity_reference: float = 0.5
    touchdown_weight: float = -0.02
    contact_on_force: float = 5.0
    contact_off_force: float = 2.0
    minimum_air_time: float = 0.01
    reset_grace_time: float = 0.1
    tread_force_ratio: float = 0.5
    wheel_radius: float = 0.09
    nearest_map_distance: float = 0.12
    approach_clearance: float = 0.06
    histogram_bins: int = 80
    histogram_max_speed: float = 4.0

    def validate(self):
        values = vars(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Quiet landing settings must be finite")
        if (self.velocity_reference <= 0 or self.touchdown_weight > 0
                or not 0 <= self.contact_off_force < self.contact_on_force
                or self.minimum_air_time <= 0 or self.reset_grace_time < 0
                or not 0 <= self.tread_force_ratio <= 1
                or self.wheel_radius <= 0 or self.nearest_map_distance <= 0
                or self.approach_clearance <= 0 or self.histogram_bins < 2
                or self.histogram_max_speed <= 0):
            raise ValueError("Invalid quiet landing settings")


def map_clearance(wheels, hits, settings):
    """Nearest original scan hit; invalid data stays invalid, never zero clearance."""
    finite = torch.isfinite(hits).all(-1)
    distance = torch.linalg.vector_norm(wheels[:, :, None, :2] - hits[:, None, :, :2], dim=-1)
    distance = torch.where(finite[:, None], distance, torch.inf)
    nearest, index = distance.min(-1)
    ground = hits[..., 2].gather(1, index)
    valid = (nearest <= settings.nearest_map_distance) & torch.isfinite(ground)
    clearance = wheels[..., 2] - ground - settings.wheel_radius
    return torch.where(valid, clearance, torch.nan), valid


class TouchdownTracker:
    """Bounded per-episode histograms, reset-safe edges, per-policy-step costs.

    Call begin_step once, then update for every physics substep with velocities
    saved BEFORE advancing physics and forces obtained AFTER advancing physics.
    Initial settling is excluded by reset_grace_time, independently per env.
    Ground/map validity is diagnostic only; real touchdowns are never map-gated.
    """
    def __init__(self, num_envs, device, settings=None):
        self.settings = settings or QuietLandingSettings()
        self.settings.validate()
        shape = (num_envs, 4)
        self.contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.initialized = torch.zeros(shape, dtype=torch.bool, device=device)
        self.air_time = torch.zeros(shape, device=device)
        self.age = torch.zeros(num_envs, device=device)
        self.last_event = torch.zeros(shape, dtype=torch.bool, device=device)
        self.last_speed = torch.zeros(shape, device=device)
        self.last_pre_velocity = torch.zeros((*shape, 3), device=device)
        self.step_cost = torch.zeros(num_envs, device=device)
        self.step_events = torch.zeros(shape, device=device)
        self.count = torch.zeros(shape, device=device)
        self.speed_sum = torch.zeros(shape, device=device)
        self.speed_sq_sum = torch.zeros(shape, device=device)
        self.speed_max = torch.zeros(shape, device=device)
        self.raw_edges = torch.zeros(shape, device=device)
        self.non_tread_edges = torch.zeros(shape, device=device)
        self.map_samples = torch.zeros(shape, device=device)
        self.valid_map_samples = torch.zeros(shape, device=device)
        self.approach_samples = torch.zeros(shape, device=device)
        self.event_in_window = torch.zeros(shape, device=device)
        self.histogram = torch.zeros((*shape, self.settings.histogram_bins), device=device)
        self.distance = torch.zeros(num_envs, device=device)

    def begin_step(self):
        self.step_cost.zero_()
        self.step_events.zero_()

    def update(self, pre_velocity, post_force, dt, clearance=None, valid=None, normals=None):
        if pre_velocity.shape != self.last_pre_velocity.shape or post_force.shape != pre_velocity.shape:
            raise ValueError("Wheel velocity and force must have shape [env, 4, 3]")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Physics dt must be positive")
        if not torch.isfinite(pre_velocity).all() or not torch.isfinite(post_force).all():
            raise ValueError("Non-finite quiet landing physics sample")
        magnitude = torch.linalg.vector_norm(post_force, dim=-1)
        contact = torch.where(self.contact, magnitude > self.settings.contact_off_force,
                              magnitude >= self.settings.contact_on_force)
        edge = contact & ~self.contact & self.initialized
        eligible = self.age[:, None] >= self.settings.reset_grace_time
        self.raw_edges += (edge & eligible).float()
        if normals is None:
            normals = torch.zeros_like(pre_velocity)
            normals[..., 2] = 1
        if normals.shape != pre_velocity.shape or not torch.isfinite(normals).all():
            raise ValueError("Invalid estimated surface normals")
        normal_force = (post_force * normals).sum(-1)
        tread = normal_force >= self.settings.tread_force_ratio * magnitude
        event = edge & tread & eligible & (self.air_time >= self.settings.minimum_air_time)
        self.non_tread_edges += (edge & ~tread & eligible).float()
        speed = (-(pre_velocity * normals).sum(-1)).clamp_min(0)
        self.last_event.copy_(event)
        self.last_pre_velocity.copy_(pre_velocity)
        self.last_speed.copy_(torch.where(event, speed, self.last_speed))
        self.step_cost += (event * (speed / self.settings.velocity_reference).square()).sum(-1)
        self.step_events += event.float()
        self.count += event.float()
        self.speed_sum += event * speed
        self.speed_sq_sum += event * speed.square()
        self.speed_max = torch.maximum(self.speed_max, torch.where(event, speed, 0.0))
        bins = (speed / self.settings.histogram_max_speed * self.settings.histogram_bins).long()
        bins = bins.clamp(0, self.settings.histogram_bins - 1)
        self.histogram.scatter_add_(-1, bins[..., None], event.float()[..., None])
        if clearance is not None and valid is not None:
            approaching = valid & torch.isfinite(clearance) & (clearance <= self.settings.approach_clearance)
            approaching &= (clearance >= -self.settings.approach_clearance) & (speed > 0) & ~self.contact
            self.map_samples += eligible.float()
            self.valid_map_samples += (valid & eligible).float()
            self.approach_samples += (approaching & eligible).float()
            self.event_in_window += (event & approaching).float()
        self.air_time = torch.where(contact, 0.0, self.air_time + dt)
        self.contact.copy_(contact)
        self.initialized.fill_(True)
        self.age += dt
        return event, speed

    def metrics(self, env_ids):
        """Only emit conditional values when samples exist; counts remain explicit."""
        result = {}
        width = self.settings.histogram_max_speed / self.settings.histogram_bins
        for leg, indices in [("all", slice(None))] + [(name, i) for i, name in enumerate(LEGS)]:
            count = self.count[env_ids, indices].sum()
            prefix = f"QuietLanding/{leg}/"
            result[prefix + "touchdown_count"] = count
            result[prefix + "raw_contact_edges"] = self.raw_edges[env_ids, indices].sum()
            result[prefix + "non_tread_edges"] = self.non_tread_edges[env_ids, indices].sum()
            result[prefix + "has_samples"] = (count > 0).float()
            samples = self.map_samples[env_ids, indices].sum()
            if samples > 0:
                result[prefix + "map_valid_fraction"] = self.valid_map_samples[env_ids, indices].sum() / samples
            if count > 0:
                result[prefix + "pre_speed_mean_m_s"] = self.speed_sum[env_ids, indices].sum() / count
                result[prefix + "pre_speed_max_m_s"] = self.speed_max[env_ids, indices].amax()
                result[prefix + "specific_energy_mean_J_kg"] = 0.5 * self.speed_sq_sum[env_ids, indices].sum() / count
                result[prefix + "height_window_event_coverage"] = self.event_in_window[env_ids, indices].sum() / count
                hist = self.histogram[env_ids, indices].reshape(-1, self.settings.histogram_bins).sum(0)
                for label, quantile in (("p50", 0.5), ("p95", 0.95)):
                    index = torch.searchsorted(hist.cumsum(0), count * quantile).clamp_max(len(hist) - 1)
                    result[prefix + f"pre_speed_{label}_hist_m_s"] = (index + 0.5) * width
                result[prefix + "histogram_tail_count"] = hist[-1]
        distance = self.distance[env_ids].sum()
        result["QuietLanding/distance_m"] = distance
        if distance > 0:
            result["QuietLanding/events_per_m"] = self.count[env_ids].sum() / distance
            result["QuietLanding/specific_energy_per_m_J_kg_m"] = 0.5 * self.speed_sq_sum[env_ids].sum() / distance
        return result

    def reset(self, env_ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[env_ids] = 0


def touchdown_cost(env):
    # Isaac's RewardManager multiplies by step_dt: cancel it for discrete events.
    return env.quiet_tracker.step_cost / env.step_dt
