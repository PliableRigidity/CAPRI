"""Commands for calibration, preview, checks, PPO train/resume/evaluate/replay."""
from __future__ import annotations
import argparse, json, math, time
from collections import Counter
from pathlib import Path
import numpy as np
from .core import AutoBotSim, load_config
from .env import AutoBotEnv, ScriptedBaseline, StopBaseline, TACTICS, ACTION_PROFILES, OPPONENT_MODES, tactic_to_drive, scripted_opponent_drive, manual_drive_command, arcade_drive
from .env import PREVIOUS_ACTION_ENCODINGS, LEGACY_PREVIOUS_ACTION_ENCODING
from .scenarios import load_scenario_file, validate_evaluation_scenarios, validate_curriculum


def load_scenarios(path):
    return validate_evaluation_scenarios(load_scenario_file(path)) if path else []


def observation_fields(profile,action_profile,previous_action_encoding=LEGACY_PREVIOUS_ACTION_ENCODING):
    slots=ACTION_PROFILES[action_profile]["previous_action_slots"]
    previous="previous_tactic_one_hot" if previous_action_encoding=="onehot_v1" else "previous_tactic_masked_zero"
    fields=["gyro_xyz/range","accel_xyz/range",f"{previous}[{slots}]"]
    if profile=="sim_debug": fields += ["own_global_x","own_global_y","sin_own_yaw","cos_own_yaw","opponent_global_x","opponent_global_y"]
    return fields


# Attribute stored on the SB3 model, so it is serialised into every checkpoint zip
# (final, latest, periodic and best) and travels with the file if it is moved.
OBSERVATION_METADATA_ATTRIBUTE="autobot_observation"


def observation_metadata(profile,action_profile,previous_action_encoding):
    return {"schema":"autobot_observation_v1","observation_profile":profile,"action_profile":action_profile,
            "previous_action_encoding":previous_action_encoding}


def _checkpoint_file(path):
    path=Path(path)
    return path if path.is_file() or path.suffix==".zip" else path.with_name(path.name+".zip")


def checkpoint_observation_metadata(path):
    """Observation metadata embedded in a checkpoint, or None if it predates versioning."""
    import zipfile
    with zipfile.ZipFile(_checkpoint_file(path)) as archive: data=json.loads(archive.read("data"))
    metadata=data.get(OBSERVATION_METADATA_ATTRIBUTE)
    return metadata if isinstance(metadata,dict) else None


def run_previous_action_encoding(path):
    """Sidecar record: effective_config.json of the run directory that owns the checkpoint."""
    checkpoint=_checkpoint_file(path)
    for directory in (checkpoint.parent,checkpoint.parent.parent):
        frozen=directory/"effective_config.json"
        if frozen.is_file(): return json.loads(frozen.read_text(encoding="utf-8")).get("previous_action_encoding")
    return None


def resolve_previous_action_encoding(checkpoint,requested=None):
    """Restore the encoding a checkpoint was trained with; refuse a conflicting request.

    Both encodings share one observation shape, so this (not a space check) is the guard
    against replaying or resuming a checkpoint under the wrong previous-action setting.
    Checkpoints saved before versioning carry no record and were all trained with onehot_v1.
    """
    embedded=(checkpoint_observation_metadata(checkpoint) or {}).get("previous_action_encoding")
    sidecar=run_previous_action_encoding(checkpoint)
    if embedded and sidecar and embedded!=sidecar:
        raise ValueError(f"{checkpoint}: checkpoint metadata says previous_action_encoding={embedded} "
                         f"but its run effective_config.json says {sidecar}")
    resolved=embedded or sidecar or LEGACY_PREVIOUS_ACTION_ENCODING
    source="checkpoint metadata" if embedded else "run effective_config.json" if sidecar else "legacy default, checkpoint predates versioning"
    if resolved not in PREVIOUS_ACTION_ENCODINGS:
        raise ValueError(f"{checkpoint}: unknown previous_action_encoding {resolved!r} ({source})")
    if requested is not None and requested!=resolved:
        raise ValueError(f"{checkpoint} was trained with previous_action_encoding={resolved} ({source}); "
                         f"refusing to run it with {requested}")
    return resolved


# PPO discount. 0.99 is the Stable-Baselines3 default that every run before this setting used.
LEGACY_GAMMA=0.99


def validate_gamma(value):
    """Discount factor in (0, 1]; raises ValueError otherwise (bools, NaN and infinities included)."""
    if isinstance(value,bool): raise ValueError(f"gamma must be a number in (0, 1], got {value!r}")
    try: gamma=float(value)
    except (TypeError,ValueError): raise ValueError(f"gamma must be a number in (0, 1], got {value!r}") from None
    if not (math.isfinite(gamma) and 0.0<gamma<=1.0): raise ValueError(f"gamma must be in (0, 1], got {value!r}")
    return gamma


def _gamma_argument(text):
    try: return validate_gamma(text)
    except ValueError as exc: raise argparse.ArgumentTypeError(str(exc)) from None


def checkpoint_ppo_settings(path):
    """PPO objective settings stored by Stable-Baselines3 inside a checkpoint zip (read without loading it)."""
    import zipfile
    with zipfile.ZipFile(_checkpoint_file(path)) as archive: data=json.loads(archive.read("data"))
    return {key:data.get(key) for key in ("gamma","gae_lambda","n_steps","batch_size","n_epochs","ent_coef","learning_rate","num_timesteps","seed")}


def resolve_gamma(requested=None,resume=None):
    """The discount PPO will use. A resume keeps the checkpoint's value and refuses a different request."""
    requested=None if requested is None else validate_gamma(requested)
    if resume is None: return LEGACY_GAMMA if requested is None else requested
    recorded=validate_gamma(checkpoint_ppo_settings(resume)["gamma"])
    if requested is not None and not math.isclose(requested,recorded,rel_tol=0.0,abs_tol=1e-12):
        raise ValueError(f"{resume} was trained with gamma={recorded}; refusing to continue it with gamma={requested} "
                         f"(that would change the objective mid-run)")
    return recorded


def ppo_hyperparameters(model):
    """The values the PPO object is actually using, read back from it rather than from the request."""
    import torch
    return {"gamma":float(model.gamma),"rollout_buffer_gamma":float(model.rollout_buffer.gamma),
            "gae_lambda":float(model.gae_lambda),"rollout_buffer_gae_lambda":float(model.rollout_buffer.gae_lambda),
            "n_steps":int(model.n_steps),
            "batch_size":int(model.batch_size),"n_epochs":int(model.n_epochs),"learning_rate":float(model.learning_rate),
            "ent_coef":float(model.ent_coef),"vf_coef":float(model.vf_coef),"clip_range":float(model.clip_range(1.0)),
            "max_grad_norm":float(model.max_grad_norm),"normalize_advantage":bool(model.normalize_advantage),
            "n_envs":int(model.n_envs),"torch_threads":int(torch.get_num_threads())}


def load_checkpoint(path,previous_action_encoding=None,device="auto"):
    """PPO.load plus the restored (and verified) previous-action encoding."""
    from stable_baselines3 import PPO
    encoding=resolve_previous_action_encoding(path,previous_action_encoding)
    model=PPO.load(path,device=device)
    metadata=dict(getattr(model,OBSERVATION_METADATA_ATTRIBUTE,None) or {})
    metadata.update(schema="autobot_observation_v1",previous_action_encoding=encoding)
    setattr(model,OBSERVATION_METADATA_ATTRIBUTE,metadata)
    return model,encoding


class CalibrationError(RuntimeError):
    """A calibration window could not be completed; the message says which limit or fault stopped it."""


CALIBRATION_MAX_PHYSICS_STEPS=200_000   # per window; 400 simulated seconds at the configured 2 ms step
CALIBRATION_MAX_WALL_SECONDS=60.0       # per window


def run_calibration_window(sim,drive,window_s,*,max_steps=CALIBRATION_MAX_PHYSICS_STEPS,
                           max_wall_seconds=CALIBRATION_MAX_WALL_SECONDS,after_chunk=None,clock=time.perf_counter):
    """Advance the calibration track by exactly `window_s` of simulated time, or raise CalibrationError.

    The physics-step count is fixed before the first step, so the loop is bounded by construction.
    The previous loop waited for `sim.data.time` to reach the window; MuJoCo resets time to zero
    when it detects an unstable step, so that loop could never finish on an unstable model.
    Returns the number of physics steps taken.
    """
    timestep=float(sim.model.opt.timestep)
    if not (math.isfinite(window_s) and window_s>0 and math.isfinite(timestep) and timestep>0):
        raise CalibrationError(f"invalid calibration window {window_s!r}s or timestep {timestep!r}s")
    total=math.ceil(window_s/timestep-1e-9)
    if total>max_steps:
        raise CalibrationError(f"step limit: a {window_s}s window needs {total} physics steps at timestep {timestep}s, limit is {max_steps}")
    started=clock(); done=0
    while done<total:
        chunk=min(sim.ctrl_steps,total-done)
        sim.step(drive,(0,0),chunk); done+=chunk
        where=f"after {done} of {total} physics steps in a {window_s}s window"
        reason=sim.instability()
        if reason: raise CalibrationError(f"{reason} ({where})")
        if abs(float(sim.data.time)-done*timestep)>timestep/2:
            raise CalibrationError(f"simulated time is {float(sim.data.time):.4f}s but {done*timestep:.4f}s was commanded; the state was reset ({where})")
        if clock()-started>max_wall_seconds:
            raise CalibrationError(f"time limit: exceeded {max_wall_seconds}s of wall-clock time ({where})")
        if after_chunk is not None: after_chunk(sim)
    return done


def calibration(args):
    import sys
    limits={"max_steps":getattr(args,"max_steps",CALIBRATION_MAX_PHYSICS_STEPS),
            "max_wall_seconds":getattr(args,"max_wall_seconds",CALIBRATION_MAX_WALL_SECONDS)}
    runs = [(2.030,3.00),(1.2192,1.81),(1.2192,1.76)]
    try:
        print("Straight-line calibration (physical target checked separately from turning):")
        print("  Method: unbounded MuJoCo calibration floor, advance tactic applied at t=0 from rest; displacement sampled at exact window end.")
        steps=0
        for measured, window in runs:
            sim=AutoBotSim(args.config, args.seed,calibration_track=True); start=sim.truth()["position_m"][:2].copy()
            # The advance tactic, not a raw (1,1): positive wheel velocity is chassis-REVERSE in this
            # model, so the old raw command compared reverse travel with forward physical runs.
            steps+=run_calibration_window(sim,tactic_to_drive(1),window,**limits)
            travelled=float(np.linalg.norm(sim.truth()["position_m"][:2]-start))
            print(f"  {window:.2f}s: simulated={travelled:.4f}m physical={measured:.4f}m target_speed=0.68m/s")
        sim=AutoBotSim(args.config, args.seed,calibration_track=True); yaw_state={"initial":sim.truth()["yaw_rad"]}
        yaw_state.update(unwrapped=yaw_state["initial"],previous=yaw_state["initial"])
        def unwrap(sim):
            yaw=sim.truth()["yaw_rad"]
            yaw_state["unwrapped"] += math.atan2(math.sin(yaw-yaw_state["previous"]), math.cos(yaw-yaw_state["previous"]))
            yaw_state["previous"]=yaw
        steps+=run_calibration_window(sim,tactic_to_drive(3),4.65,after_chunk=unwrap,**limits)
        turned=abs(yaw_state["unwrapped"]-yaw_state["initial"]); rotations=turned/(2*math.pi); rate=turned/4.65
        print(f"Turn-in-place 4.65s: simulated={rotations:.3f} rotations, yaw_rate={rate:.3f}rad/s; physical target=8.25 rotations, 11.1rad/s")
        print("390 wheel RPM is an inferred no-slip starting estimate, not a measured motor speed.")
        print(f"Calibration completed: {len(runs)+1} windows, {steps} physics steps, no instability reported.")
    except CalibrationError as exc:
        # Fail loudly and stop: never spin on a model MuJoCo keeps resetting.
        print(f"CALIBRATION FAILED: {exc}",file=sys.stderr)
        raise SystemExit(2)


def check(args):
    import mujoco
    from gymnasium.utils.env_checker import check_env
    env=AutoBotEnv(args.config, "hardware_plausible", args.seed); check_env(env, skip_render_check=True)
    masked=AutoBotEnv(args.config,"hardware_plausible",args.seed,previous_action_encoding="masked_zero_v1")
    check_env(masked,skip_render_check=True)
    masked_reset,_=masked.reset(seed=123); masked_step=masked.step(1)[0]; plain_reset,_=env.reset(seed=123)
    assert masked.observation_space==env.observation_space and masked_reset.shape==plain_reset.shape
    assert not masked_reset[6:11].any() and not masked_step[6:11].any() and plain_reset[6]==1.0
    a,_=env.reset(seed=123); b,_=env.reset(seed=123)
    assert np.array_equal(a,b) and a.shape==(11,)
    debug=AutoBotEnv(args.config,"sim_debug",args.seed); assert debug.reset(seed=123)[0].shape==(17,)
    # Fixed sequence determinism without rendering.
    def trajectory(render=False):
        s=AutoBotSim(args.config,123); out=[]
        renderer=mujoco.Renderer(s.model,height=120,width=160) if render else None
        for action in [1]*10+[3]*8+[0]*4:
            out.append(s.step(tactic_to_drive(action))["position_m"].tolist())
            if renderer is not None: renderer.update_scene(s.data,camera="overhead"); renderer.render()
        if renderer is not None: renderer.close()
        return np.asarray(out)
    assert np.array_equal(trajectory(False),trajectory(False))
    assert np.array_equal(trajectory(False),trajectory(True))
    print("PASS: Gymnasium checker, seeded reset, observation separation, shapes, and render-independent trajectory determinism")


def wall_clock():
    """Wall-time source of the 2D viewer loop (a function, so headless checks can substitute a controlled clock)."""
    return time.perf_counter()


class RealTimePacer:
    """How many fixed control steps the viewer runs this frame so simulated time tracks wall time.

    The simulation step itself never changes (one control step); only how many of them run per rendered frame
    does, so the pace no longer depends on the render rate. Time spent paused, at a reset or after a round end
    is not owed. After a stall at most `max_steps_per_frame` steps run and the rest of the backlog is dropped,
    so play slows down instead of jumping ahead.
    """
    def __init__(self,step_s,max_steps_per_frame=5):
        self.step_s=float(step_s); self.max_steps_per_frame=int(max_steps_per_frame); self.last=None; self.owed=0.0

    def hold(self):
        self.last=None; self.owed=0.0

    def due(self,now):
        if self.last is None: self.owed=self.step_s           # starting or resuming: one step now, then follow the clock
        else: self.owed+=max(0.0,now-self.last)
        self.last=now
        steps=int((self.owed+1e-9)//self.step_s)
        if steps>self.max_steps_per_frame: steps=self.max_steps_per_frame; self.owed=steps*self.step_s
        self.owed=max(0.0,self.owed-steps*self.step_s)
        return steps


def blue_controller_label(path):
    """What drives blue, for the HUD. A supervised clone's run says so, and its checkpoint has no PPO timesteps."""
    checkpoint=_checkpoint_file(path)
    for directory in (checkpoint.parent,checkpoint.parent.parent):
        sidecar=directory/"effective_config.json"
        if sidecar.is_file():
            if str(json.loads(sidecar.read_text(encoding="utf-8")).get("training","")).upper().startswith("SUPERVISED"):
                return "Supervised clone"
            break
    return "Supervised clone" if checkpoint_ppo_settings(checkpoint)["num_timesteps"]==0 else "PPO"


def session_log_path(requested,started=None):
    """A new input-log file for every play session: the requested name with the session's start time added."""
    requested=Path(requested); suffix=requested.suffix or ".jsonl"
    stamp=time.strftime("%Y%m%d-%H%M%S",time.localtime(started))
    candidate=requested.with_name(f"{requested.stem}-{stamp}{suffix}"); count=1
    while candidate.exists():
        count+=1; candidate=requested.with_name(f"{requested.stem}-{stamp}-{count}{suffix}")
    return candidate


class ReplaySession:
    """One checkpoint/scenario replay loop shared by every viewer frontend."""
    def __init__(self,args):
        scenarios=load_scenarios(args.eval_scenarios)
        selected=[scenarios[args.scenario_index % len(scenarios)]] if scenarios else None
        # Restored from the checkpoint/run metadata; an explicit conflicting flag is an error.
        self.model,self.previous_action_encoding=load_checkpoint(args.model,getattr(args,"previous_action_encoding",None))
        self.env=AutoBotEnv(args.config,args.profile,args.seed,opponent_model=args.opponent_chassis,
                            opponent_mode=args.opponent or "scripted",evaluation_scenarios=selected,
                            reward_profile=args.reward_profile,
                            collision_overlay=getattr(args,"collision_overlay",False),
                            action_profile=getattr(args,"action_profile","legacy5"),
                            previous_action_encoding=self.previous_action_encoding)
        self.sim=self.env.sim; self.seed=args.seed
        if self.model.observation_space!=self.env.observation_space or self.model.action_space!=self.env.action_space:
            raise ValueError(f"Checkpoint spaces {self.model.observation_space}/{self.model.action_space} are incompatible with action profile {self.env.action_profile} spaces {self.env.observation_space}/{self.env.action_space}")
        self.tactic_hold_s=max(0.0,float(getattr(args,"tactic_hold_ms",0.0))/1000)
        self.current_tactic="stop"; self.requested_tactic="stop"; self.held_action=0; self.hold_until_s=0.0
        self.end_reason=None; self.reset()
        print(f"Replay observation setting: previous_action_encoding={self.previous_action_encoding}")

    def reset(self):
        self.env.reset(seed=self.seed); self.sim.set_drive(0,0,0); self.sim.set_drive(1,0,0)
        self.current_tactic="stop"; self.requested_tactic="stop"; self.held_action=0; self.hold_until_s=0.0
        self.end_reason=None

    def step(self,opponent_drive_override=None,forced_action=None):
        requested=int(self.model.predict(self.env._obs(),deterministic=True)[0]) if forced_action is None else int(forced_action)
        self.requested_tactic=self.env.tactics[requested]
        if self.tactic_hold_s<=0 or self.sim.data.time+1e-12>=self.hold_until_s:
            self.held_action=requested; self.hold_until_s=float(self.sim.data.time)+self.tactic_hold_s
        action=self.held_action
        self.current_tactic=self.env.tactics[action]
        _,_,terminated,truncated,info=self.env.step(action,opponent_drive_override)
        info["requested_tactic"]=self.requested_tactic; info["applied_tactic"]=self.current_tactic
        if terminated or truncated:
            reason=info["round_end_reason"]
            self.end_reason="Timeout (no loss)" if reason=="Timeout" else (
                f"{reason} (bot {info['round_end_bot']})" if info["round_end_bot"] is not None else reason)
            self.sim.set_drive(0,0,0); self.sim.set_drive(1,0,0)
        return terminated or truncated,info


def preview(args):
    try: import pygame
    except ImportError as exc: raise SystemExit("Install pygame to use the 2D preview") from exc
    replay_env = None; model = None; replay_session=None
    if getattr(args, "model", None):
        replay_session=ReplaySession(args); replay_env=replay_session.env
        sim=replay_session.sim; model=replay_session.model
    else: sim=AutoBotSim(args.config,args.seed,opponent_model=args.opponent_chassis)
    human_red=bool(getattr(args,"human_red",False)); replay_commands=[]; replay_index=0
    # Every play session writes its own file: --input-log names it and the start time is added.
    input_log=session_log_path(args.input_log) if human_red and getattr(args,"input_log",None) else None
    blue_label=blue_controller_label(args.model) if replay_session is not None else None
    if human_red and getattr(args,"red_input_replay",None):
        replay_commands=[json.loads(line) for line in Path(args.red_input_replay).read_text(encoding="utf-8").splitlines()
                         if json.loads(line).get("type")=="step"]
    if input_log:
        input_log.parent.mkdir(parents=True,exist_ok=True); input_log.touch(exist_ok=False)
        args.resolved_input_log=str(input_log); print(f"Input log for this session: {input_log}")
    pygame.init()
    screen=pygame.display.set_mode((args.width,args.height),pygame.RESIZABLE)
    pygame.display.set_caption("AutoBot MuJoCo top-down preview")
    clock=pygame.time.Clock(); font=pygame.font.SysFont(None,21)
    drive=(0,0); paused=True; just_reset=True; round_end=None; reset_held=False; current_tactic="stop"
    # Simulated time follows wall time: the fixed control step is unchanged, the number of steps per frame varies.
    pacer=RealTimePacer(1.0/sim.cfg["simulation"]["control_hz"])
    opponent_mode=args.opponent or sim.cfg["opponent"]["mode"]

    def reset_round():
        nonlocal paused,just_reset,round_end,drive,current_tactic,replay_index
        if replay_session is not None: replay_session.reset()
        else: sim.reset()
        sim.set_drive(0,0,0); sim.set_drive(1,0,0)
        drive=(0,0); current_tactic="stop"; replay_index=0; paused=True; just_reset=True; round_end=None
        if input_log:
            with input_log.open("a",encoding="utf-8") as handle: handle.write(json.dumps({"type":"reset","time_s":0.0})+"\n")

    def draw_frame():
        """Render MuJoCo state only; this function never advances or alters physics."""
        width,height=screen.get_size(); hud_h=96; padding=24
        arena=sim.cfg["arena"]; arena_w=arena["width_m"]; arena_h=arena["height_m"]
        available_w=max(1,width-2*padding); available_h=max(1,height-hud_h-2*padding)
        scale=min(available_w/arena_w,available_h/arena_h)
        draw_w=arena_w*scale; draw_h=arena_h*scale
        left=(width-draw_w)/2; top=hud_h+padding+(available_h-draw_h)/2
        arena_rect=pygame.Rect(round(left),round(top),round(draw_w),round(draw_h))
        def pt(xy): return (round(left+xy[0]*scale),round(top+(arena_h-xy[1])*scale))

        background=(25,29,34); floor=(221,224,220); boundary=(105,112,118); pit=(12,15,18)
        screen.fill(background)
        pygame.draw.rect(screen,floor,arena_rect)
        # A circle centred exactly on a corner contributes only its in-arena quarter.
        # Clipping makes each quarter-circle the playable-floor edge and prevents HUD spill.
        old_clip=screen.get_clip(); screen.set_clip(arena_rect)
        radius=max(1,round(arena["pit_radius_m"]*scale))
        for centre in arena["pit_centres_m"]:
            centre_px=pt(centre)
            pygame.draw.circle(screen,pit,centre_px,radius)
            pygame.draw.circle(screen,(225,25,25),centre_px,radius,max(2,round(arena["pit_rim_width_m"]*scale)))
        screen.set_clip(old_clip)
        pygame.draw.rect(screen,boundary,arena_rect,2)

        flipper=[pt(v) for v in arena["flipper_vertices_m"]]
        pygame.draw.polygon(screen,(229,174,61),flipper)
        pygame.draw.polygon(screen,(104,70,12),flipper,3)

        for i,color in ((0,(35,130,245)),(1,(240,65,50))):
            robot=sim.robot_config(i); local=np.asarray(robot["outline_m"],dtype=float)
            truth=sim.truth(i); yaw=truth["yaw_rad"]
            rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            centre=truth["position_m"][:2]
            polygon=[pt(centre+rot@point) for point in local]
            pygame.draw.polygon(screen,color,polygon); pygame.draw.polygon(screen,(245,245,245),polygon,2)
            # Wheel markers follow the same configured spacing and yaw.
            wheel_length=max(0.018,robot["wheel_diameter_m"]*.75)
            for wheel_world in sim.wheel_positions(i):
                wheel_centre=wheel_world[:2]
                wheel_a=wheel_centre+rot@np.array([-wheel_length/2,0]); wheel_b=wheel_centre+rot@np.array([wheel_length/2,0])
                pygame.draw.line(screen,(20,20,22),pt(wheel_a),pt(wheel_b),max(2,round(.012*scale)))
            front=centre+rot@np.array([float(local[:,0].max()),0])
            pygame.draw.circle(screen,(255,240,120),pt(front),max(2,round(.006*scale)))

        truth=sim.truth()
        state=f"ROUND END: {round_end}" if round_end else ("PAUSED - SPACE to start" if paused else "RUNNING")
        elapsed=min(float(sim.data.time),float(sim.cfg["simulation"]["episode_seconds"]))
        end_reason=round_end or "none"
        controls="W/S throttle + A/D steer red | SPACE pause | R reset" if human_red else "WASD/arrows drive | SPACE start/pause | R reset | close window quit"
        controllers=f"blue={blue_label}, red=human" if human_red and not replay_commands else f"blue={blue_label}, red=recording" if human_red else f"opponent={opponent_mode}/{sim.robot_model_key(1)}"
        lines=[controls,
               f"{state} | {controllers} | tactic={current_tactic} | elapsed={elapsed:.2f}/{sim.cfg['simulation']['episode_seconds']:.2f}s | end={end_reason}",
               f"pose=({truth['position_m'][0]:.3f}, {truth['position_m'][1]:.3f}) yaw={truth['yaw_rad']:.2f} | red rim is non-colliding"]
        for index,line in enumerate(lines): screen.blit(font.render(line,True,(235,235,235)),(padding,14+index*25))
        return arena_rect

    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
            if e.type==pygame.KEYDOWN and e.key==pygame.K_r and not reset_held: reset_round(); reset_held=True
            if e.type==pygame.KEYDOWN and e.key==pygame.K_SPACE and round_end is None: paused=not paused
        keys=pygame.key.get_pressed()
        # Polling fallback makes reset reliable even if a KEYDOWN event is missed. One press resets once:
        # the key event and the poll share `reset_held`, and R must be released before it resets again.
        if keys[pygame.K_r] and not reset_held: reset_round()
        reset_held=bool(keys[pygame.K_r])
        direction="forward" if keys[pygame.K_UP] or keys[pygame.K_w] else "reverse" if keys[pygame.K_DOWN] or keys[pygame.K_s] else "left" if keys[pygame.K_LEFT] or keys[pygame.K_a] else "right" if keys[pygame.K_RIGHT] or keys[pygame.K_d] else "stop"
        drive=manual_drive_command(direction)
        # Always render the exact reset state once before the first physics step.
        if not paused and not just_reset and round_end is None: steps_due=pacer.due(wall_clock())
        else: steps_due=0; pacer.hold()
        for _ in range(steps_due):
            if round_end is not None: break
            if replay_session is not None:
                red_override=None
                if human_red:
                    if replay_commands:
                        red_override=tuple(replay_commands[min(replay_index,len(replay_commands)-1)]["red_command"])
                    elif pygame.key.get_focused():
                        red_override=arcade_drive(keys[pygame.K_w] or keys[pygame.K_UP],keys[pygame.K_s] or keys[pygame.K_DOWN],
                                                  keys[pygame.K_a] or keys[pygame.K_LEFT],keys[pygame.K_d] or keys[pygame.K_RIGHT])
                    else: red_override=(0.0,0.0)
                    replay_index+=1
                done,step_info=replay_session.step(red_override); current_tactic=replay_session.current_tactic
                if input_log:
                    record={"type":"step","time_s":float(sim.data.time),"blue_tactic":current_tactic,
                            "blue_command":list(step_info["blue_drive"]),"red_command":list(step_info["opponent_drive"]),
                            "end_reason":replay_session.end_reason}
                    with input_log.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record)+"\n")
                if done:
                    round_end=replay_session.end_reason; paused=True
            else:
                current_tactic=direction
                opponent_drive=(0,0) if opponent_mode=="static" else scripted_opponent_drive(sim)
                sim.step(drive,opponent_drive)
                event=sim.arena_event()
                if event is not None:
                    round_end=f"{event.reason} (bot {event.bot})"
                    sim.set_drive(0,0,0); sim.set_drive(1,0,0); paused=True
                elif sim.data.time >= sim.cfg["simulation"]["episode_seconds"]:
                    round_end="Timeout (no loss)"
                    sim.set_drive(0,0,0); sim.set_drive(1,0,0); paused=True
        draw_frame(); pygame.display.flip()
        if args.screenshot:
            output=Path(args.screenshot); output.parent.mkdir(parents=True,exist_ok=True)
            pygame.image.save(screen,str(output)); print(f"Saved reset-state screenshot: {output.resolve()}"); running=False
        just_reset=False; clock.tick(60)
    pygame.quit()


def replay3d(args):
    """Native MuJoCo frontend over the exact same ReplaySession as Pygame."""
    import mujoco
    import mujoco.viewer
    session=ReplaySession(args); state={"paused":True,"just_reset":True}
    def key(code):
        if code==32 and session.end_reason is None: state["paused"]=not state["paused"]
        if code in (82,114): session.reset(); state.update(paused=True,just_reset=True)
    with mujoco.viewer.launch_passive(session.sim.model,session.sim.data,key_callback=key) as viewer:
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid=mujoco.mj_name2id(session.sim.model,mujoco.mjtObj.mjOBJ_CAMERA,"overhead")
        while viewer.is_running():
            started=time.perf_counter()
            if not state["paused"] and not state["just_reset"] and session.end_reason is None:
                done,_=session.step()
                if done: state["paused"]=True
            elapsed=min(float(session.sim.data.time),float(session.sim.cfg["simulation"]["episode_seconds"]))
            status="ROUND END" if session.end_reason else ("PAUSED" if state["paused"] else "RUNNING")
            viewer.set_texts((None,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "SPACE start/pause | R reset",
                f"{status}\ntactic={session.current_tactic}\nelapsed={elapsed:.2f}/{session.sim.cfg['simulation']['episode_seconds']:.2f}s\nend={session.end_reason or 'none'}"))
            viewer.sync(); state["just_reset"]=False
            time.sleep(max(0,1/session.sim.cfg["simulation"]["control_hz"]-(time.perf_counter()-started)))


def replay(args):
    return replay3d(args) if args.viewer=="3d" else preview(args)


def preview3d(args):
    """Native MuJoCo view of the exact same model used headlessly and by 2D."""
    import mujoco.viewer
    sim=AutoBotSim(args.config,args.seed,opponent_model=args.opponent_chassis,collision_overlay=args.collision_overlay); command=[0.0,0.0]
    def key(code):
        mapping={265:(1,1),264:(-.65,-.65),263:(-.7,.7),262:(.7,-.7),32:(0,0)}
        if code in mapping: command[:]=mapping[code]
        if code in (82,114): sim.reset()
    with mujoco.viewer.launch_passive(sim.model,sim.data,key_callback=key) as viewer:
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED; viewer.cam.fixedcamid=mujoco.mj_name2id(sim.model,mujoco.mjtObj.mjOBJ_CAMERA,args.view)
        while viewer.is_running():
            start=time.perf_counter(); sim.step(command); viewer.sync(); time.sleep(max(0,1/sim.cfg['simulation']['control_hz']-(time.perf_counter()-start)))


def snapshot3d(args):
    """Save a native MuJoCo camera image for alignment review."""
    import mujoco
    from PIL import Image
    sim=AutoBotSim(args.config,args.seed,opponent_model=args.opponent_chassis,collision_overlay=args.collision_overlay)
    renderer=mujoco.Renderer(sim.model,height=args.height,width=args.width)
    renderer.update_scene(sim.data,camera=args.view)
    image=renderer.render()
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(image).save(output); renderer.close()
    print(f"Saved MuJoCo {args.view} view: {output.resolve()}")


def evaluate_model(model,config,profile,seed,opponent,opponent_chassis,scenarios,reward_profile="outcome_only",action_profile="legacy5",
                   previous_action_encoding=None):
    """Deterministic fixed-suite metrics shared by CLI evaluation and checkpoint selection."""
    totals=Counter(); actions=Counter(); groups={}; episodes=[]
    if previous_action_encoding is None:
        # Checkpoints carry their own setting; scripted baselines and legacy checkpoints do not.
        previous_action_encoding=(getattr(model,OBSERVATION_METADATA_ATTRIBUTE,None) or {}).get(
            "previous_action_encoding",LEGACY_PREVIOUS_ACTION_ENCODING)
    for episode,scenario in enumerate(scenarios):
        group=scenario.get("group","ungrouped")
        group_metrics=groups.setdefault(group,Counter())
        env=AutoBotEnv(config,profile,seed+episode,opponent_model=opponent_chassis,
                       opponent_mode=opponent,reward_profile=reward_profile,action_profile=action_profile,
                       previous_action_encoding=previous_action_encoding)
        obs,_=env.reset(seed=seed+episode,options={"scenario":scenario}); stationary=0; total=0.0
        first_contact=None; switches=0; previous_action=None; opponent_start=env.sim.truth(1)["position_m"][:2].copy()
        blue_start=env.sim.truth(0)["position_m"][:2].copy(); blue_previous=blue_start.copy()
        pre_contact_path=0.0; pre_contact_position=None
        reset_probabilities=None
        if hasattr(model,"policy"):
            tensor,_=model.policy.obs_to_tensor(obs)
            reset_probabilities={name:float(value) for name,value in zip(env.tactics,
                model.policy.get_distribution(tensor).distribution.probs.detach().cpu().numpy()[0])}
        while True:
            action=int(model.predict(obs,deterministic=True)[0]); actions[env.tactics[action]]+=1
            obs,reward,terminated,truncated,info=env.step(action); total+=reward
            blue_position=env.sim.truth(0)["position_m"][:2].copy()
            if first_contact is None: pre_contact_path += float(np.linalg.norm(blue_position-blue_previous))
            blue_previous=blue_position
            if previous_action is not None: switches+=int(action!=previous_action)
            previous_action=action
            if first_contact is None and info["opponent_contact"]:
                first_contact=float(env.sim.data.time); pre_contact_position=blue_position.copy()
            stationary += int(info["stationary_wall_contact"])
            if terminated or truncated: break
        if info["round_end_reason"]=="Pit fall": outcome="own_falls" if info["round_end_bot"]==0 else "wins"
        elif info["round_end_reason"]=="Simultaneous pit fall": outcome="draws"
        elif info["round_end_reason"]=="Timeout": outcome="timeouts"
        else: outcome="simulation_bugs"
        totals[outcome]+=1; totals["stationary_wall_steps"]+=stationary; group_metrics[outcome]+=1
        opponent_displacement=float(np.linalg.norm(env.sim.truth(1)["position_m"][:2]-opponent_start))
        timeout_class=None
        if outcome=="timeouts":
            timeout_class="never_reached_red" if first_contact is None else "contact_stall" if opponent_displacement<0.05 else "moved_red_no_finish"
        episodes.append({"scenario":scenario["name"],"group":group,"outcome":outcome,"return":total,
                         "time_to_first_contact_s":first_contact,"tactic_switches":switches,
                         "switches_per_s":switches/max(float(env.sim.data.time),1e-9),
                         "pre_contact_net_displacement_m":float(np.linalg.norm(
                             (pre_contact_position if pre_contact_position is not None else blue_position)-blue_start)),
                         "pre_contact_path_m":pre_contact_path,
                         "opponent_displacement_m":opponent_displacement,"timeout_class":timeout_class,
                         "stationary_wall_steps":stationary,
                         "stationary_wall_time_s":stationary/env.sim.cfg["simulation"]["control_hz"],
                         "reset_action_probabilities":reset_probabilities})
    outcome_keys=("wins","own_falls","draws","timeouts","simulation_bugs")
    return {"episodes":episodes,"outcomes":{key:totals[key] for key in outcome_keys},
            "groups":{name:{key:counts[key] for key in outcome_keys} for name,counts in groups.items()},
            "action_counts":{name:actions[name] for name in ACTION_PROFILES[action_profile]["tactics"]},
            "stationary_wall_steps":totals["stationary_wall_steps"]}


def make_training_callbacks(out,args,scenarios,completed_episodes=0,resumed_stage=None):
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    class DiagnosticsCallback(BaseCallback):
        def __init__(self):
            super().__init__(); self.started=time.perf_counter(); self.counts=[Counter() for _ in range(args.n_envs)]
            self.stationary=[0]*args.n_envs; self.metrics_path=out/"metrics.jsonl"
            self.first_contact=[None]*args.n_envs; self.episode_start=[None]*args.n_envs
            self.episode_index=int(completed_episodes); self.stage=resumed_stage
        def _on_step(self):
            actions=np.asarray(self.locals.get("actions",[])).reshape(-1)
            infos=self.locals.get("infos",[]); dones=np.asarray(self.locals.get("dones",[])).reshape(-1)
            for index,action in enumerate(actions):
                if self.episode_start[index] is None: self.episode_start[index]=self.num_timesteps-len(actions)
                self.counts[index][ACTION_PROFILES[args.action_profile]["tactics"][int(action)]]+=1
                if index<len(infos): self.stationary[index]+=int(infos[index].get("stationary_wall_contact",False))
                if index<len(infos) and self.first_contact[index] is None and infos[index].get("opponent_contact"):
                    self.first_contact[index]=sum(self.counts[index].values())
                if index<len(dones) and dones[index]:
                    info=infos[index]; reason=info.get("round_end_reason")
                    stage=info.get("curriculum_stage")
                    if stage!=self.stage:
                        # Logged from what the environment actually sampled, not inferred from the step count.
                        change={"type":"curriculum_stage_change","episode_index":self.episode_index,
                                "first_episode_started_at_timesteps":self.episode_start[index],
                                "timesteps":self.num_timesteps,"from":self.stage,"to":stage}
                        with self.metrics_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(change)+"\n")
                        print(f"Curriculum stage {self.stage} -> {stage} at episode {self.episode_index} (timesteps {self.num_timesteps})")
                        self.stage=stage
                    outcome="own_fall" if reason=="Pit fall" and info.get("round_end_bot")==0 else "win" if reason=="Pit fall" and info.get("round_end_bot")==1 else "draw" if reason=="Simultaneous pit fall" else "timeout" if reason=="Timeout" else "simulation_bug"
                    record={"type":"train_episode","timesteps":self.num_timesteps,"environment":index,
                            "stage":info.get("curriculum_stage"),"scenario_group":info.get("scenario_group"),
                            "outcome":outcome,"actions":dict(self.counts[index]),
                            "stationary_wall_steps":self.stationary[index],
                            "episode_index":self.episode_index,"scenario":info.get("scenario"),
                            "steps":sum(self.counts[index].values()),"first_contact_step":self.first_contact[index]}
                    with self.metrics_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record)+"\n")
                    self.logger.record(f"outcomes/{outcome}",1); self.counts[index]=Counter(); self.stationary[index]=0
                    self.first_contact[index]=None; self.episode_start[index]=self.num_timesteps; self.episode_index+=1
            if self.num_timesteps and self.num_timesteps % max(1,args.latest_freq*args.n_envs) < args.n_envs:
                self.model.save(out/"latest_model")
            if args.wall_clock_hours and time.perf_counter()-self.started >= args.wall_clock_hours*3600:
                self.model.save(out/"latest_model")
                with self.metrics_path.open("a",encoding="utf-8") as handle:
                    handle.write(json.dumps({"type":"wall_clock_stop","timesteps":self.num_timesteps})+"\n")
                return False
            return True
    class OutcomeEvaluationCallback(BaseCallback):
        def __init__(self):
            super().__init__(); self.metrics_path=out/"metrics.jsonl"; selection_path=out/"best"/"selection.json"
            self.best_score=tuple(json.loads(selection_path.read_text())["selection_score"]) if selection_path.exists() else None
        def _on_step(self):
            if self.n_calls % max(1,args.eval_freq//args.n_envs): return True
            result=evaluate_model(self.model,args.config,args.profile,args.seed+10000,args.opponent,
                                  args.opponent_chassis,scenarios,args.reward_profile,args.action_profile,
                                  args.previous_action_encoding)
            outcomes=result["outcomes"]
            # Documented lexicographic match metric: maximize wins, then minimize
            # own falls, draws, and timeouts. Wall time and shaped return never select best.
            score=(outcomes["wins"],-outcomes["own_falls"],-outcomes["draws"],-outcomes["timeouts"])
            status="tie"
            if self.best_score is None or score>self.best_score:
                status="initial" if self.best_score is None else "improved"
                self.best_score=score; (out/"best").mkdir(exist_ok=True); self.model.save(out/"best"/"best_model")
                (out/"best"/"selection.json").write_text(json.dumps({"timesteps":self.num_timesteps,
                    "selection_score":score,"metric":"maximize wins, then minimize own falls, draws, timeouts"},indent=2))
            record={"type":"evaluation","timesteps":self.num_timesteps,"selection_status":status,
                    "selection_score":score,"metrics":result}
            with self.metrics_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record)+"\n")
            self.logger.record("eval/wins",outcomes["wins"]); self.logger.record("eval/own_falls",outcomes["own_falls"])
            self.logger.record("eval/timeouts",outcomes["timeouts"])
            return True
    return [CheckpointCallback(max(1,args.checkpoint_freq//args.n_envs),str(out/"checkpoints")),
            DiagnosticsCallback(),OutcomeEvaluationCallback()]


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    out=Path(args.output)
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise SystemExit(f"Refusing to overwrite non-empty fresh run directory: {out}")
    if args.resume:
        # Resume restores the checkpoint's own setting; an explicit conflicting flag is refused.
        try: encoding=resolve_previous_action_encoding(args.resume,getattr(args,"previous_action_encoding",None))
        except ValueError as exc: raise SystemExit(f"Resume configuration mismatch: {exc}")
    else: encoding=getattr(args,"previous_action_encoding",None) or LEGACY_PREVIOUS_ACTION_ENCODING
    args.previous_action_encoding=encoding
    try: gamma=resolve_gamma(getattr(args,"gamma",None),args.resume)
    except ValueError as exc: raise SystemExit(f"{'Resume configuration mismatch' if args.resume else 'Invalid training configuration'}: {exc}")
    torch_threads=getattr(args,"torch_threads",None)
    if torch_threads is not None:
        if int(torch_threads)<1: raise SystemExit("Invalid training configuration: --torch-threads must be at least 1")
        import torch; torch.set_num_threads(int(torch_threads))
    out.mkdir(parents=True,exist_ok=True); cfg=load_config(args.config)
    scenarios=load_scenarios(args.eval_scenarios); curriculum=validate_curriculum(load_scenario_file(args.training_curriculum))
    metadata={"config":cfg,"config_path":str(Path(args.config).resolve()),"seed":args.seed,"evaluation_seed":args.seed+10000,
              "profile":args.profile,"n_envs":args.n_envs,"opponent_mode":args.opponent,"opponent_chassis":args.opponent_chassis,
              "reward_profile":args.reward_profile,"action_profile":args.action_profile,
              "previous_action_encoding":encoding,"gamma":gamma,
              "action_names":list(ACTION_PROFILES[args.action_profile]["tactics"]),"observation_normalization":False,
              "observation_fields":observation_fields(args.profile,args.action_profile,encoding),"evaluation_scenarios":scenarios,
              "training_curriculum":curriculum}
    frozen_path=out/"effective_config.json"
    if args.resume and frozen_path.exists():
        frozen=json.loads(frozen_path.read_text(encoding="utf-8"))
        keys=("profile","n_envs","opponent_mode","opponent_chassis","reward_profile","action_profile","observation_normalization",
              "previous_action_encoding","gamma")
        # Run directories written before a setting was versioned used its legacy value.
        legacy={"action_profile":"legacy5","previous_action_encoding":LEGACY_PREVIOUS_ACTION_ENCODING,"gamma":LEGACY_GAMMA}
        mismatches=[key for key in keys if frozen.get(key,legacy.get(key))!=metadata.get(key)]
        if frozen.get("config")!=cfg: mismatches.append("config")
        if frozen.get("evaluation_scenarios")!=scenarios: mismatches.append("evaluation_scenarios")
        if frozen.get("training_curriculum")!=curriculum: mismatches.append("training_curriculum")
        if mismatches: raise SystemExit(f"Resume configuration mismatch: {', '.join(mismatches)}")
        write_frozen=False
    else: write_frozen=True
    import yaml
    if not args.resume:
        (out/"effective_sim.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
        (out/"evaluation_scenarios.json").write_text(json.dumps(scenarios,indent=2),encoding="utf-8")
        (out/"training_curriculum.json").write_text(json.dumps(curriculum,indent=2),encoding="utf-8")
    completed_episodes=0; resumed_stage=None
    if args.resume and (out/"metrics.jsonl").exists():
        previous=[json.loads(line) for line in (out/"metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        previous=[row for row in previous if row.get("type")=="train_episode"]
        completed_episodes=len(previous); resumed_stage=previous[-1].get("stage") if previous else None
    env=make_vec_env(lambda: AutoBotEnv(args.config,args.profile,opponent_model=args.opponent_chassis,
                         opponent_mode=args.opponent,training_curriculum=curriculum,reward_profile=args.reward_profile,
                         curriculum_start_episode=completed_episodes,action_profile=args.action_profile,
                         previous_action_encoding=encoding),n_envs=args.n_envs,seed=args.seed)
    if args.resume:
        model=PPO.load(args.resume,env=env,device=args.device,tensorboard_log=str(out/"tensorboard"))
    else: model=PPO("MlpPolicy",env,device=args.device,seed=args.seed,verbose=1,n_steps=min(256,max(8,args.timesteps//2)),batch_size=32,gamma=gamma,tensorboard_log=str(out/"tensorboard"))
    # Record what PPO is really using. A mismatch here would mean the run optimises a different objective than labelled.
    actual=ppo_hyperparameters(model)
    if not all(math.isclose(actual[key],gamma,rel_tol=0.0,abs_tol=1e-12) for key in ("gamma","rollout_buffer_gamma")):
        raise SystemExit(f"PPO is using gamma={actual['gamma']} (rollout buffer {actual['rollout_buffer_gamma']}) "
                         f"but gamma={gamma} was resolved for this run")
    metadata["ppo_hyperparameters"]=actual
    if write_frozen: frozen_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    with (out/"metrics.jsonl").open("a",encoding="utf-8") as handle:
        handle.write(json.dumps({"type":"run_start","resumed_from":args.resume,"starting_timesteps":int(model.num_timesteps),
                                 "previous_action_encoding":encoding,"action_profile":args.action_profile,"profile":args.profile,
                                 "ppo_hyperparameters":actual})+"\n")
    # Stored in model.__dict__, so SB3 writes it into every checkpoint this run saves.
    setattr(model,OBSERVATION_METADATA_ATTRIBUTE,observation_metadata(args.profile,args.action_profile,encoding))
    callbacks=make_training_callbacks(out,args,scenarios,completed_episodes,resumed_stage); started=time.perf_counter(); before=model.num_timesteps
    model.learn(args.timesteps,callback=callbacks,reset_num_timesteps=not bool(args.resume),tb_log_name="ppo")
    elapsed=time.perf_counter()-started; model.save(out/"latest_model"); model.save(out/"final_model")
    summary={"type":"run_complete","starting_timesteps":before,"ending_timesteps":model.num_timesteps,
             "elapsed_seconds":elapsed,"environment_steps_per_second":(model.num_timesteps-before)/elapsed}
    with (out/"metrics.jsonl").open("a",encoding="utf-8") as handle: handle.write(json.dumps(summary)+"\n")
    print(json.dumps(summary,indent=2)); print(f"Saved {out/'final_model.zip'} and resumable {out/'latest_model.zip'}")


def evaluate(args):
    requested=getattr(args,"previous_action_encoding",None)
    model,encoding=load_checkpoint(args.model,requested) if args.model else (ScriptedBaseline(),requested)
    scenarios=load_scenarios(args.eval_scenarios)
    if args.episodes is not None: scenarios=[scenarios[index%len(scenarios)] for index in range(args.episodes)]
    result=evaluate_model(model,args.config,args.profile,args.seed,args.opponent,args.opponent_chassis,
                          scenarios,args.reward_profile,args.action_profile,encoding)
    result["reward_profile"]=args.reward_profile; print(json.dumps(result,indent=2))


def benchmark(args):
    scenarios=load_scenarios(args.scenarios); policies={"always_stop":StopBaseline(),"always_advance":ScriptedBaseline()}
    for specification in args.model:
        # Each checkpoint is evaluated under its own restored previous-action encoding.
        name,path=specification.split("=",1); policies[name]=load_checkpoint(path,device="cpu")[0]
    results={}
    for policy_name,policy in policies.items():
        results[policy_name]={}
        for behavior in args.opponent_behavior:
            results[policy_name][behavior]=evaluate_model(policy,args.config,args.profile,args.seed,
                behavior,args.opponent_chassis,scenarios,args.reward_profile,args.action_profile)
    output={"seed":args.seed,"config":str(Path(args.config).resolve()),"scenarios":scenarios,"results":results}
    if args.output:
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/sim.yaml"); p.add_argument("--seed",type=int,default=7)
    sub=p.add_subparsers(dest="command",required=True)
    cal=sub.add_parser("calibrate"); cal.add_argument("--max-steps",type=int,default=CALIBRATION_MAX_PHYSICS_STEPS,help="physics-step limit per calibration window"); cal.add_argument("--max-wall-seconds",type=float,default=CALIBRATION_MAX_WALL_SECONDS,help="wall-clock limit per calibration window")
    sub.add_parser("check")
    view=sub.add_parser("preview"); view.add_argument("--width",type=int,default=900); view.add_argument("--height",type=int,default=900); view.add_argument("--screenshot"); view.add_argument("--opponent",choices=["static","scripted"]); view.add_argument("--opponent-chassis",choices=["last_year","autobot"],default="last_year")
    view3=sub.add_parser("preview3d"); view3.add_argument("--opponent-chassis",choices=["last_year","autobot"],default="last_year"); view3.add_argument("--view",choices=["overhead","side"],default="overhead"); view3.add_argument("--collision-overlay",action="store_true")
    shot=sub.add_parser("snapshot3d"); shot.add_argument("--opponent-chassis",choices=["last_year","autobot"],default="last_year"); shot.add_argument("--view",choices=["overhead","side"],default="overhead"); shot.add_argument("--collision-overlay",action="store_true"); shot.add_argument("--width",type=int,default=1000); shot.add_argument("--height",type=int,default=700); shot.add_argument("--output",required=True)
    t=sub.add_parser("train"); t.add_argument("--timesteps",type=int,default=100000); t.add_argument("--output",default="runs/ppo"); t.add_argument("--resume"); t.add_argument("--device",default="cpu"); t.add_argument("--n-envs",type=int,default=1); t.add_argument("--profile",choices=["hardware_plausible","sim_debug"],default="hardware_plausible"); t.add_argument("--action-profile",choices=ACTION_PROFILES,default="legacy5"); t.add_argument("--opponent",choices=OPPONENT_MODES,default="scripted"); t.add_argument("--opponent-chassis",choices=["last_year","autobot"],default="last_year"); t.add_argument("--reward-profile",choices=["outcome_only"],default="outcome_only"); t.add_argument("--training-curriculum",default="config/training_curriculum.json"); t.add_argument("--checkpoint-freq",type=int,default=10000); t.add_argument("--latest-freq",type=int,default=2500); t.add_argument("--eval-freq",type=int,default=5000); t.add_argument("--eval-episodes",type=int,default=6); t.add_argument("--eval-scenarios",default="config/eval_scenarios.json"); t.add_argument("--wall-clock-hours",type=float); t.add_argument("--gamma",type=_gamma_argument,help="PPO discount in (0, 1]; default 0.99 for a fresh run, the checkpoint's value on resume (a different value is refused)"); t.add_argument("--torch-threads",type=int,help="explicit Torch intra-op thread count for this process (recorded in the run metadata)"); t.add_argument("--previous-action-encoding",choices=PREVIOUS_ACTION_ENCODINGS,help="default: onehot_v1 for a fresh run; otherwise restored from the checkpoint")
    e=sub.add_parser("evaluate"); e.add_argument("--model"); e.add_argument("--episodes",type=int); e.add_argument("--profile",choices=["hardware_plausible","sim_debug"],default="hardware_plausible"); e.add_argument("--action-profile",choices=ACTION_PROFILES,default="legacy5"); e.add_argument("--opponent",choices=OPPONENT_MODES,default="scripted"); e.add_argument("--opponent-chassis",choices=["last_year","autobot"],default="last_year"); e.add_argument("--reward-profile",choices=["outcome_only"],default="outcome_only"); e.add_argument("--eval-scenarios",default="config/eval_scenarios.json"); e.add_argument("--previous-action-encoding",choices=PREVIOUS_ACTION_ENCODINGS,help="default: onehot_v1 for a fresh run; otherwise restored from the checkpoint")
    r=sub.add_parser("replay"); r.add_argument("--model",required=True); r.add_argument("--viewer",choices=["2d","3d"],default="2d"); r.add_argument("--collision-overlay",action="store_true"); r.add_argument("--tactic-hold-ms",type=float,default=0.0); r.add_argument("--profile",choices=["hardware_plausible","sim_debug"],default="hardware_plausible"); r.add_argument("--action-profile",choices=ACTION_PROFILES,default="legacy5"); r.add_argument("--width",type=int,default=900); r.add_argument("--height",type=int,default=900); r.add_argument("--screenshot"); r.add_argument("--opponent",choices=OPPONENT_MODES); r.add_argument("--opponent-chassis",choices=["last_year","autobot"],default="last_year"); r.add_argument("--reward-profile",choices=["outcome_only"],default="outcome_only"); r.add_argument("--eval-scenarios"); r.add_argument("--scenario-index",type=int,default=0); r.add_argument("--previous-action-encoding",choices=PREVIOUS_ACTION_ENCODINGS,help="default: onehot_v1 for a fresh run; otherwise restored from the checkpoint"); r.set_defaults(human_red=False)
    play=sub.add_parser("play"); play.add_argument("--model",required=True); play.add_argument("--viewer",choices=["2d"],default="2d"); play.add_argument("--collision-overlay",action="store_true"); play.add_argument("--tactic-hold-ms",type=float,default=0.0); play.add_argument("--profile",choices=["hardware_plausible","sim_debug"],default="sim_debug"); play.add_argument("--action-profile",choices=ACTION_PROFILES,default="legacy5"); play.add_argument("--width",type=int,default=900); play.add_argument("--height",type=int,default=900); play.add_argument("--screenshot"); play.add_argument("--opponent",choices=OPPONENT_MODES,default="static"); play.add_argument("--opponent-chassis",choices=["last_year","autobot"],default="last_year"); play.add_argument("--reward-profile",choices=["outcome_only"],default="outcome_only"); play.add_argument("--eval-scenarios",required=True); play.add_argument("--scenario-index",type=int,default=2); play.add_argument("--input-log",default="artifacts/human-vs-ppo-input.jsonl"); play.add_argument("--red-input-replay"); play.add_argument("--previous-action-encoding",choices=PREVIOUS_ACTION_ENCODINGS,help="default: onehot_v1 for a fresh run; otherwise restored from the checkpoint"); play.set_defaults(human_red=True)
    b=sub.add_parser("benchmark"); b.add_argument("--model",action="append",required=True,help="NAME=checkpoint.zip"); b.add_argument("--scenarios",required=True); b.add_argument("--opponent-behavior",action="append",choices=OPPONENT_MODES,required=True); b.add_argument("--profile",choices=["hardware_plausible","sim_debug"],default="sim_debug"); b.add_argument("--action-profile",choices=ACTION_PROFILES,default="legacy5"); b.add_argument("--opponent-chassis",choices=["last_year","autobot"],default="last_year"); b.add_argument("--reward-profile",choices=["outcome_only"],default="outcome_only"); b.add_argument("--output")
    args=p.parse_args(argv); return {"calibrate":calibration,"check":check,"preview":preview,"preview3d":preview3d,"snapshot3d":snapshot3d,"replay":replay,"play":preview,"train":train,"evaluate":evaluate,"benchmark":benchmark}[args.command](args)

if __name__ == "__main__": main()
