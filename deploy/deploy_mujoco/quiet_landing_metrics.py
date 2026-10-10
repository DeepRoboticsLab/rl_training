"""Quiet V1 MuJoCo adapter: shared event accounting, real contact force samples."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/rl_training"))
from m20_quiet.signals import LEGS, QuietLandingSettings, TouchdownTracker, map_clearance
from m20_quiet.extended import ExtendedQuietSettings, ExtendedQuietTracker, surface_normals


class QuietMujocoMetrics:
    def __init__(self, model, contract, settings=None):
        if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("Split-step quiet diagnostics cannot preserve RK4 integration")
        self.model, self.contract = model, contract
        self.kinematics = mujoco.MjData(model)
        self.tracker = TouchdownTracker(1, "cpu", settings or QuietLandingSettings())
        self.settings = self.tracker.settings
        self.extended = ExtendedQuietTracker(1, "cpu", ExtendedQuietSettings())
        self.command_xy = np.zeros(2)
        self.records = {name: [] for name in ("time", "clearance", "map_valid", "pre_velocity",
                                              "contact", "event", "speed", "specific_energy", "force")}
        self.events = []
        self.extended_records = {}
        self.last_event_time = np.full(4, -np.inf)
        self.wheel_masses = model.body_mass[contract.wheel_body_ids].copy()
        self.current_velocity = np.zeros((4, 3))
        self.current_force = np.zeros((4, 3))
        self.force_max = np.zeros(4)
        self.translation_energy = np.zeros(4)
        for name in ("wheel_velocity_w", "wheel_speed_m_s", "wheel_translation_energy_J", "force_norm_N", "root_pos_w"):
            self.records[name] = []

    def pre_step(self, data):
        velocity = np.empty((4, 3))
        for i, body in enumerate(self.contract.wheel_body_ids):
            spatial = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, int(body), spatial, 0)
            velocity[i] = spatial[3:]
        wheels = torch.as_tensor(data.xpos[self.contract.wheel_body_ids].copy())[None]
        hits = torch.as_tensor(self.contract.last_scan_hits.copy())[None]
        clearance, valid = map_clearance(wheels, hits, self.settings)
        normals, normal_valid = surface_normals(wheels, hits, self.extended.settings)
        return torch.as_tensor(velocity, dtype=torch.float32)[None], clearance, valid, normals, normal_valid

    def refresh_post_kinematics(self, data):
        self.kinematics.qpos[:] = data.qpos
        self.kinematics.qvel[:] = data.qvel
        self.kinematics.time = data.time
        mujoco.mj_fwdPosition(self.model, self.kinematics)
        mujoco.mj_fwdVelocity(self.model, self.kinematics)
        return self.kinematics

    def post_step(self, data, snapshot, record=True):
        forces = np.zeros((4, 3))
        wheel_index = {int(body): i for i, body in enumerate(self.contract.wheel_body_ids)}
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if contact.efc_address < 0:
                continue
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            # Only robot/static-terrain contacts, not leg/self contacts.
            if not ((body1 in wheel_index and body2 == 0) or (body2 in wheel_index and body1 == 0)):
                continue
            local = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, contact_id, local)
            world = contact.frame.reshape(3, 3).T @ local[:3]
            if body1 in wheel_index:
                forces[wheel_index[body1]] -= world
            if body2 in wheel_index:
                forces[wheel_index[body2]] += world
        velocity, clearance, valid, normals, normal_valid = snapshot
        event, speed = self.tracker.update(velocity, torch.as_tensor(forces, dtype=torch.float32)[None],
                                           self.model.opt.timestep, clearance, valid, normals)
        post = np.zeros((4, 6))
        # mj_step2 integrates qpos/qvel but leaves cvel/xpos at PRE-step values.
        # Refresh only a scratch data object; preserve this step's solved forces,
        # joint acceleration and actuator force on the original data object.
        self.refresh_post_kinematics(data)
        for i, body in enumerate(self.contract.wheel_body_ids):
            mujoco.mj_objectVelocity(self.model, self.kinematics, mujoco.mjtObj.mjOBJ_BODY, int(body), post[i], 0)
        self.current_velocity = post[:, 3:].copy()
        self.current_force = forces.copy()
        self.force_max = np.maximum(self.force_max, np.linalg.norm(forces, axis=-1))
        self.translation_energy = .5*self.wheel_masses*np.sum(self.current_velocity**2, axis=-1)
        base = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.kinematics, mujoco.mjtObj.mjOBJ_BODY, self.contract.base_id, base, 0)
        tensor = lambda value: torch.as_tensor(value.copy(), dtype=torch.float32)[None]
        hits = self.contract.last_scan_hits
        heights = hits[np.isfinite(hits).all(-1), 2]
        stair = len(heights) > 0 and np.ptp(heights) > self.extended.settings.stair_height_threshold
        swing = (velocity*normals).sum(-1) > .05
        if not (bool(stair) and bool(np.linalg.norm(self.command_xy) > .1)):
            swing.zero_()
        self.extended.update(
            event=event, contact=self.tracker.contact,
            eligible=self.tracker.age >= self.settings.reset_grace_time,
            force=tensor(forces), normals=normals.float(),
            wheel_velocity=tensor(post[:, 3:]), wheel_omega=tensor(post[:, :3]), radius=self.settings.wheel_radius,
            wheel_acceleration=tensor(data.qacc[self.contract.qvel_ids[12:]]),
            base_omega=tensor(base[:3]), base_velocity=tensor(base[3:]),
            torque=tensor(data.actuator_force[self.contract.actuator_ids]),
            clearance=clearance, map_valid=valid, swing_gate=swing, dt=self.model.opt.timestep)
        for i in range(4):
            if event[0, i]:
                self.last_event_time[i] = data.time
                self.events.append({"time": float(data.time), "leg": LEGS[i],
                                    "pre_speed_m_s": float(speed[0, i]),
                                    "specific_energy_J_kg": float(0.5 * speed[0, i] ** 2),
                                    "map_valid": bool(valid[0, i])})
        if record:
            for key, value in self.extended.last.items():
                self.extended_records.setdefault(key, []).append(value[0].numpy().copy())
            self.extended_records.setdefault("normal_fit_valid", []).append(normal_valid[0].numpy().copy())
            values = (float(data.time), clearance[0].numpy().copy(), valid[0].numpy().copy(),
                      velocity[0].numpy().copy(), self.tracker.contact[0].numpy().copy(),
                      event[0].numpy().copy(), speed[0].numpy().copy(),
                      (0.5 * speed[0].square()).numpy(), forces.copy())
            for key, value in zip(("time", "clearance", "map_valid", "pre_velocity", "contact", "event", "speed", "specific_energy", "force"), values):
                self.records[key].append(value)
            for key, value in (("wheel_velocity_w", self.current_velocity),
                               ("wheel_speed_m_s", np.linalg.norm(self.current_velocity, axis=-1)),
                               ("wheel_translation_energy_J", self.translation_energy),
                               ("force_norm_N", np.linalg.norm(forces, axis=-1)),
                               ("root_pos_w", self.kinematics.xpos[self.contract.base_id])):
                self.records[key].append(value.copy())

    def draw(self, viewer, data):
        # Caller draws height scan first (it clears user_scn).
        with viewer.lock():
            scene = viewer.user_scn
            for i, wheel in enumerate(self.contract.wheel_body_ids):
                if scene.ngeom + 3 > scene.maxgeom:
                    raise RuntimeError("MuJoCo user scene lacks space for quiet landing overlays")
                recent = data.time - self.last_event_time[i] < 0.15
                contact = bool(self.tracker.contact[0, i])
                color = np.array([1., .15, .1, 1.] if recent else
                                 [.2, .9, .2, 1.] if contact else [.25, .55, 1., 1.], dtype=np.float32)
                position = self.kinematics.xpos[wheel].copy()
                position[2] += 0.12
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.array([.025, 0., 0.]), position, np.eye(3).ravel(), color)
                geom.label = LEGS[i].upper()
                scene.ngeom += 1
                for vector, scale, rgba in ((self.current_velocity[i], .15, [.2, .7, 1., 1.]),
                                             (self.current_force[i], .001, [1., .65, .1, 1.])):
                    length = np.linalg.norm(vector)*scale
                    if length < 1e-6:
                        continue
                    delta = vector*scale*min(1., .45/length)
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                                       np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .012,
                                        position, position+delta)
                    scene.ngeom += 1
        if hasattr(viewer, "set_texts"):
            viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              "Quiet V1\n" + self.table_text(), ""))

    def table_text(self):
        lines = ["Wheel center velocity [world m/s] | wheel translation E [J] | terrain force [N]",
                 "Leg State      vx     vy     vz  |v|    Etr     |F|    Fz   Fmax   TDvn  TD[J/kg]"]
        for i, leg in enumerate(LEGS):
            state = "support" if self.tracker.contact[0, i] else "air"
            speed = float(self.tracker.last_speed[0, i])
            v = self.current_velocity[i]
            f = self.current_force[i]
            values = f"{speed:5.2f} {0.5*speed*speed:8.3f}" if np.isfinite(self.last_event_time[i]) else "   --       --"
            lines.append(f"{leg.upper():2}  {state:7} {v[0]:6.2f} {v[1]:6.2f} {v[2]:6.2f} "
                         f"{np.linalg.norm(v):4.2f} {self.translation_energy[i]:6.3f} "
                         f"{np.linalg.norm(f):7.1f} {f[2]:6.1f} {self.force_max[i]:6.1f} {values}")
        lines += ["Arrows: blue=velocity (0.15m per m/s), orange=force (0.001m per N).",
                  "Arrow length capped at 0.45m. Fmax: since start.",
                  "TD: last pre-impact normal speed / energy per mass. Etr: wheel-only proxy."]
        return "\n".join(lines)

    def summary(self):
        result = {key: float(value) for key, value in self.tracker.metrics(slice(None)).items()}
        result.update({"QuietCost/"+key: float(value[0]) for key, value in self.extended.costs.items()})
        result.update({key: float(value) for key, value in self.extended.metrics(slice(None)).items()})
        return result

    def save(self, path, metadata):
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{key: np.asarray(value) for key, value in self.records.items()},
                            **{key: np.asarray(value) for key, value in self.extended_records.items()},
                            metadata=json.dumps({**metadata, "wheel_masses_kg": self.wheel_masses.tolist(),
                                                 "energy_definition": "0.5 * wheel_body_mass * post_step_center_speed_squared; translation only"}))
        path.with_suffix(".events.json").write_text(json.dumps(self.events, indent=2))
        self.plot(path.with_suffix(".png"))
        self.plot_extended(path.with_name(path.stem+"_extended.png"))

    def plot_extended(self, path):
        import matplotlib.pyplot as plt
        if not self.records["time"]:
            return
        keys = ("normal_force_N", "window_excess_impulse_N_s", "loading_rate_N_s", "slip_m_s",
                "base_angular_acceleration_rad_s2", "base_vertical_acceleration_m_s2")
        fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
        for ax, key in zip(axes, keys):
            values = np.asarray(self.extended_records[key])
            labels = (LEGS if values.ndim == 2 and values.shape[1] == 4 else
                      ("world x", "world y", "world z") if values.ndim == 2 else ("world z",))
            if values.ndim == 1:
                ax.plot(self.records["time"], values, label=labels[0])
            else:
                for i, label in enumerate(labels):
                    ax.plot(self.records["time"], values[:, i], label=label.upper())
            ax.legend(loc="upper right", fontsize=7)
            ax.set_ylabel(key, fontsize=8)
            ax.grid(alpha=.2)
        axes[-1].set_xlabel("simulation time [s]")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)

    def plot(self, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.records["time"]:
            return
        fig, axes = plt.subplots(4, 3, figsize=(17, 10), sharex=True)
        times = np.asarray(self.records["time"])
        velocity = np.asarray(self.records["speed"])
        events = np.asarray(self.records["event"])
        values = (np.asarray(self.records['wheel_speed_m_s']),
                  np.asarray(self.records['wheel_translation_energy_J']), np.asarray(self.records['force_norm_N']))
        for i in range(4):
            for j, unit in enumerate(('speed [m/s]', 'wheel translation E [J]', 'terrain force [N]')):
                ax = axes[i,j]
                ax.plot(times, values[j][:,i], label=unit)
                if j == 0:
                    ax.plot(times, velocity[:,i], alpha=.7, label='pre normal closing speed')
                    ax.scatter(times[events[:,i]],velocity[events[:,i],i],color='red',s=12,label='touchdown')
                ax.set_ylabel(LEGS[i].upper())
                ax.legend(fontsize=7)
                ax.grid(alpha=.2)
        for ax in axes[-1]: ax.set_xlabel("simulation time [s]")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)


class LiveSignals:
    """One figure, four wheels by three physically separate units."""
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, self.axes = plt.subplots(4, 3, sharex=True, figsize=(14, 9))
        self.lines = []
        self.normal_lines = []
        for i in range(4):
            row = []
            for j, unit in enumerate(('speed [m/s]', 'wheel translation E [J]', 'terrain force [N]')):
                ax = self.axes[i,j]
                row.append(ax.plot([], [], label=unit)[0])
                if j == 0:
                    self.normal_lines.append(ax.plot([], [], label='pre normal closing [m/s]', color='tab:red', alpha=.7)[0])
                ax.set_ylabel(LEGS[i].upper())
                ax.legend(fontsize=7)
            self.lines.append(row)
        for ax in self.axes[-1]: ax.set_xlabel('simulation time [s]')
        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, metrics):
        times = np.asarray(metrics.records['time'][-1000:])
        if times.size == 0: return
        for j, key in enumerate(('wheel_speed_m_s','wheel_translation_energy_J','force_norm_N')):
            values = np.asarray(metrics.records[key][-1000:])
            for i in range(4):
                self.lines[i][j].set_data(times,values[:,i])
                if j == 0:
                    normal = np.asarray(metrics.records['speed'][-1000:])
                    self.normal_lines[i].set_data(times,normal[:,i])
                self.axes[i,j].relim(); self.axes[i,j].autoscale_view()
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

    def close(self):
        self.plt.close(self.fig)
