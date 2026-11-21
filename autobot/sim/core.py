"""Shared deterministic MuJoCo core; viewers and Gymnasium read this same state."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml
from .arena import classify_sim_state, floor_triangles, pit_rim_segments


def load_config(path: str | Path = "config/sim.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _vec(values) -> str:
    return " ".join(str(float(x)) for x in values)


@dataclass(frozen=True)
class ImuReading:
    gyro_rad_s: np.ndarray
    accel_m_s2: np.ndarray
    timestamp_s: float


# MuJoCo raises these when a step produces NaN/Inf/huge values. By default it then resets the
# whole state (time back to 0) and carries on, so a loop that waits for simulated time to reach
# a target never ends. The counters survive that automatic reset, which makes them detectable.
INSTABILITY_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                        mujoco.mjtWarning.mjWARN_BADQACC)


class AutoBotSim:
    """Two differential-drive robots in one arena; bot 0 is policy controlled."""

    def __init__(self, config_path="config/sim.yaml", seed: int | None = None,
                 opponent_model: str | None = None, collision_overlay: bool = False,
                 calibration_track: bool = False):
        self.config_path = str(config_path)
        self.cfg = load_config(config_path)
        if opponent_model is not None:
            if opponent_model not in self.cfg["robots"]: raise ValueError(f"Unknown opponent model: {opponent_model}")
            self.cfg["models"]["opponent"] = opponent_model
        self.collision_overlay=collision_overlay
        self.calibration_track=calibration_track
        self.rng = np.random.default_rng(self.cfg["simulation"]["seed"] if seed is None else seed)
        self.model = mujoco.MjModel.from_xml_string(self._xml(),assets=self._mesh_assets())
        self.data = mujoco.MjData(self.model)
        self.ctrl_steps = round(1 / self.cfg["simulation"]["control_hz"] / self.model.opt.timestep)
        self._last_imu_time = -math.inf
        self._last_imu = None
        self.reset()

    def robot_config(self,bot: int) -> dict:
        key=self.cfg["models"]["player" if bot==0 else "opponent"]
        return self.cfg["robots"][key]

    def robot_model_key(self,bot: int) -> str:
        return self.cfg["models"]["player" if bot==0 else "opponent"]

    def _mesh_assets(self):
        assets={}
        for bot in (0,1):
            path=Path(self.robot_config(bot)["visual_mesh"]["path"])
            if path.is_file(): assets[path.name]=path.read_bytes()
        return assets

    def _xml(self) -> str:
        c, a = self.cfg, self.cfg["arena"]
        friction=self.cfg["physics"]["friction"]
        root = ET.Element("mujoco", model="autobot_arena")
        ET.SubElement(root, "option", timestep=str(c["simulation"]["timestep_s"]), gravity="0 0 -9.81", integrator="RK4")
        ET.SubElement(root, "compiler", angle="degree")
        visual_settings=ET.SubElement(root,"visual")
        ET.SubElement(visual_settings,"global",offwidth="1200",offheight="900")
        default = ET.SubElement(root, "default")
        ET.SubElement(default, "joint", damping="0.002", armature="0.00002")
        asset=ET.SubElement(root,"asset")
        floor_meshes=[]; depth=float(a["floor_thickness_m"])
        for index,triangle in enumerate([] if self.calibration_track else floor_triangles(a)):
            name=f"floor_triangle_{index}"; floor_meshes.append(name)
            vertices=[]
            for z in (-depth,0.0):
                for point in triangle: vertices.extend([point[0],point[1],z])
            faces=[0,2,1,3,4,5,0,1,4,0,4,3,1,2,5,1,5,4,2,0,3,2,3,5]
            ET.SubElement(asset,"mesh",name=name,vertex=_vec(vertices),face=" ".join(map(str,faces)))
        for bot in (0,1):
            visual=self.robot_config(bot)["visual_mesh"]; path=Path(visual["path"])
            if path.is_file():
                ET.SubElement(asset,"mesh",name=f"bot{bot}_visual_asset",file=path.name,scale=_vec(visual["scale"]))
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="0.575 0.575 2", dir="0 0 -1")
        if self.calibration_track:
            ET.SubElement(world,"geom",name="calibration_floor",type="plane",size="5 5 0.02",rgba="0.12 0.14 0.16 1",friction=_vec(friction))
        else:
            for name in floor_meshes:
                ET.SubElement(world,"geom",name=name,type="mesh",mesh=name,rgba="0.12 0.14 0.16 1",friction=_vec(friction))
            thickness=float(a["wall_thickness_m"]); height=float(a["wall_height_m"]); w=float(a["width_m"]); h=float(a["height_m"])
            wall_specs=(("bottom",[w/2,-thickness/2,height/2],[w/2+thickness,thickness/2,height/2]),
                        ("top",[w/2,h+thickness/2,height/2],[w/2+thickness,thickness/2,height/2]),
                        ("left",[-thickness/2,h/2,height/2],[thickness/2,h/2+thickness,height/2]),
                        ("right",[w+thickness/2,h/2,height/2],[thickness/2,h/2+thickness,height/2]))
            for side,pos,size in wall_specs:
                ET.SubElement(world,"geom",name=f"wall_{side}",type="box",pos=_vec(pos),size=_vec(size),rgba="0.3 0.32 0.35 1",friction=_vec(friction))
            for index,(start,end) in enumerate(pit_rim_segments(a)):
                ET.SubElement(world,"geom",name=f"pit_rim_{index}",type="capsule",
                              fromto=_vec([start[0],start[1],0.004,end[0],end[1],0.004]),
                              size=str(a["pit_rim_width_m"]/2),rgba="0.9 0.03 0.03 1",
                              contype="0",conaffinity="0",group="1")
        ET.SubElement(world, "camera", name="overhead", pos="0.575 0.575 1.65", xyaxes="1 0 0 0 1 0", mode="fixed")
        ET.SubElement(world, "camera", name="side", pos="0.05 0.575 0.08", xyaxes="0 -1 0 0 0 1", fovy="70", mode="fixed")
        for i in range(2):
            self._robot_xml(world, i)
        actuators = ET.SubElement(root, "actuator")
        for i in range(2):
            r=self.robot_config(i); gain=str(r["drive_gain"]); max_speed=self.max_wheel_rad_s(i)
            for side in ("l", "r"):
                ET.SubElement(actuators, "velocity", name=f"bot{i}_{side}_motor", joint=f"bot{i}_{side}_wheel", kv=gain, ctrllimited="true", ctrlrange=f"{-max_speed} {max_speed}")
        return ET.tostring(root, encoding="unicode")

    def max_wheel_rad_s(self,bot=0):
        return self.robot_config(bot)["max_wheel_rpm"] * 2 * math.pi / 60

    def _robot_xml(self, world, index):
        r = self.robot_config(index)
        sx, sy, sz = r["chassis_size_m"]
        radius = r["wheel_diameter_m"] / 2
        half_track = r["wheel_spacing_m"] / 2; axle_x=r["rear_axle_offset_m"]
        body = ET.SubElement(world, "body", name=f"bot{index}", pos=f"0 0 {radius + sz/2}")
        ET.SubElement(body, "freejoint", name=f"bot{index}_free")
        # Collision geometry is deliberately independent from any future visual STL.
        collision_rgba="1 0.1 0.1 0.75" if self.collision_overlay else "0 0 0 0"
        ET.SubElement(body, "geom", name=f"bot{index}_chassis_collision", type="box", size=_vec([sx/2, sy/2, sz/2]), mass=str(r["mass_kg"]), rgba=collision_rgba)
        visual=r["visual_mesh"]
        if Path(visual["path"]).is_file():
            alpha="0.35" if self.collision_overlay else "1"
            color=f"0.12 0.45 0.95 {alpha}" if index==0 else f"0.92 0.18 0.12 {alpha}"
            ET.SubElement(body,"geom",name=f"bot{index}_visual",type="mesh",mesh=f"bot{index}_visual_asset",
                          pos=_vec(visual["centre_offset_m"]),euler=_vec(visual["euler_deg"]),rgba=color,
                          contype="0",conaffinity="0",mass="0",group="2")
        for side, y in (("l", half_track), ("r", -half_track)):
            wheel = ET.SubElement(body, "body", name=f"bot{index}_{side}_wheel_body", pos=f"{axle_x} {y} {-sz/2}", euler="90 0 0")
            ET.SubElement(wheel, "joint", name=f"bot{index}_{side}_wheel", type="hinge", axis="0 0 1")
            wheel_rgba="0.1 1 0.1 0.85" if self.collision_overlay else "0 0 0 0"
            ET.SubElement(wheel, "geom", name=f"bot{index}_{side}_wheel_collision",type="cylinder", size=_vec([radius, 0.006]), mass=str(r["wheel_mass_kg"]), friction=_vec(self.cfg["physics"]["friction"]), rgba=wheel_rgba)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        arena = self.cfg["arena"]
        inset, x = arena["spawn_inset_m"], arena["width_m"] / 2
        for i, (y, yaw) in enumerate(((inset, math.pi/2), (arena["height_m"]-inset, -math.pi/2))):
            r=self.robot_config(i); radius=r["wheel_diameter_m"]/2
            qadr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"bot{i}_free")]
            self.data.qpos[qadr:qadr+3] = [x, y, radius + r["chassis_size_m"][2]/2]
            self.data.qpos[qadr+3:qadr+7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
        if self.calibration_track:
            self.set_pose(0,0.0,0.0,0.0)
            self.set_pose(1,-2.0,-2.0,0.0)
        mujoco.mj_forward(self.model, self.data)
        self._last_imu_time, self._last_imu = -math.inf, None
        return self.truth()

    def set_pose(self, bot: int, x: float, y: float, yaw: float):
        """Test/setup helper; normal controllers must use set_drive/step instead."""
        qadr=self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f"bot{bot}_free")]
        self.data.qpos[qadr:qadr+2]=[x,y]
        self.data.qpos[qadr+3:qadr+7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        self.data.qvel[self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f"bot{bot}_free")]:][:6]=0
        mujoco.mj_forward(self.model,self.data)

    def arena_event(self):
        return classify_sim_state([self.truth(0)["position_m"],self.truth(1)["position_m"]],self.cfg["arena"])

    def set_drive(self, bot: int, left: float, right: float):
        self.data.ctrl[bot*2:bot*2+2] = np.clip([left, right], -1, 1) * self.max_wheel_rad_s(bot)

    def step(self, bot0=(0.0, 0.0), bot1=(0.0, 0.0), control_steps: int | None = None):
        self.set_drive(0, *bot0); self.set_drive(1, *bot1)
        for _ in range(self.ctrl_steps if control_steps is None else control_steps):
            mujoco.mj_step(self.model, self.data)
        return self.truth()

    def instability(self) -> str | None:
        """Why the current state can no longer be trusted, or None. Read-only; never alters physics."""
        flagged=[f"{warning.name} x{int(self.data.warning[warning].number)} (last DOF {int(self.data.warning[warning].lastinfo)})"
                 for warning in INSTABILITY_WARNINGS if self.data.warning[warning].number]
        if flagged:
            return "MuJoCo reported an unstable simulation and reset the state: "+", ".join(flagged)
        for name in ("qpos","qvel","qacc"):
            if not np.isfinite(getattr(self.data,name)).all(): return f"non-finite {name}"
        return None

    def truth(self, bot=0):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"bot{bot}")
        quat = self.data.xquat[bid]
        mat = np.empty(9); mujoco.mju_quat2Mat(mat, quat)
        yaw = math.atan2(mat[3], mat[0])
        return {"time_s": float(self.data.time), "position_m": self.data.xpos[bid].copy(), "yaw_rad": yaw,
                "linear_velocity_m_s": self.data.cvel[bid][3:6].copy(), "angular_velocity_rad_s": self.data.cvel[bid][:3].copy()}

    def wheel_positions(self,bot=0):
        result=[]
        for side in ("l","r"):
            bid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f"bot{bot}_{side}_wheel_body")
            result.append(self.data.xpos[bid].copy())
        return result

    def imu(self, bot=0) -> ImuReading:
        """Noisy sensor-frame measurement; never returns pose/heading/opponent state."""
        ic = self.cfg["imu"]
        if self._last_imu is not None and self.data.time - self._last_imu_time < 1/ic["sample_rate_hz"]:
            return self._last_imu
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"bot{bot}")
        # World-to-body rotation; mounting XYZ Euler rotation is applied afterward.
        rot = self.data.xmat[bid].reshape(3, 3)
        angles = np.radians(ic["mounting_euler_deg"])
        cx,cy,cz=np.cos(angles); sx,sy,sz=np.sin(angles)
        rm=np.array([[cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz],[cy*sz,sx*sy*sz+cx*cz,cx*sy*sz-sx*cz],[-sy,sx*cy,cx*cy]])
        gyro = rm.T @ (rot.T @ self.data.cvel[bid][:3])
        # Proper acceleration at rest is +g along body up; sufficient for settled inversion.
        accel = rm.T @ (rot.T @ np.array([0.0, 0.0, 9.81]))
        gyro += np.asarray(ic["gyro_bias_rad_s"]) + self.rng.normal(0, ic["gyro_noise_std_rad_s"], 3)
        accel += np.asarray(ic["accel_bias_m_s2"]) + self.rng.normal(0, ic["accel_noise_std_m_s2"], 3)
        gyro = np.clip(gyro, -ic["gyro_range_rad_s"], ic["gyro_range_rad_s"])
        accel = np.clip(accel, -ic["accel_range_m_s2"], ic["accel_range_m_s2"])
        self._last_imu = ImuReading(gyro, accel, float(self.data.time)); self._last_imu_time = self.data.time
        return self._last_imu
