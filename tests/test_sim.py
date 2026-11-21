import numpy as np
import math
import mujoco
from autobot.sim.env import AutoBotEnv, tactic_to_drive, scripted_opponent_drive, manual_drive_command, arcade_drive, opponent_drive
from autobot.sim.core import AutoBotSim

def _bots_touch(sim):
    for index in range(sim.data.ncon):
        contact=sim.data.contact[index]
        names=[mujoco.mj_id2name(sim.model,mujoco.mjtObj.mjOBJ_GEOM,int(geom)) or ""
               for geom in (contact.geom1,contact.geom2)]
        if any(name.startswith("bot0_") for name in names) and any(name.startswith("bot1_") for name in names):
            return True
    return False

def test_observation_profiles_do_not_leak_truth():
    standard=AutoBotEnv(); debug=AutoBotEnv(observation_profile="sim_debug")
    assert standard.reset(seed=1)[0].shape == (11,)
    assert debug.reset(seed=1)[0].shape == (17,)
    experimental5=AutoBotEnv(observation_profile="sim_debug",action_profile="experimental5_v1")
    arc7=AutoBotEnv(observation_profile="sim_debug",action_profile="arc7_v1")
    assert experimental5.reset(seed=1)[0].shape==(19,) and experimental5.action_space.n==5
    assert arc7.reset(seed=1)[0].shape==(19,) and arc7.action_space.n==7

def test_sim_debug_distinguishes_mirrored_pit_geometries():
    env=AutoBotEnv(observation_profile="sim_debug",seed=1)
    top={"name":"top","player":[0.353553,0.796447,2.356194],"opponent":[0.226274,0.923726,2.356194]}
    bottom={"name":"bottom","player":[0.796447,0.353553,-0.785398],"opponent":[0.923726,0.226274,-0.785398]}
    top_obs,_=env.reset(seed=1,options={"scenario":top})
    bottom_obs,_=env.reset(seed=1,options={"scenario":bottom})
    assert not np.array_equal(top_obs[11:],bottom_obs[11:])
    assert not np.allclose(top_obs[11:13],bottom_obs[11:13])
    assert not np.allclose(top_obs[15:17],bottom_obs[15:17])

def test_seeded_curriculum_is_reproducible_and_logs_stage_and_group():
    def sample():
        env=AutoBotEnv(seed=77,training_curriculum="config/training_curriculum.json")
        result=[]
        for index in range(5):
            _,info=env.reset(seed=77 if index==0 else None)
            result.append((info["scenario"],info["curriculum_stage"],info["scenario_group"]))
        return result
    assert sample()==sample()
    assert all(stage=="near_pit" for _,stage,_ in sample())

def test_configured_models_and_rear_axles_align_after_ninety_degree_turn():
    sim=AutoBotSim(seed=1)
    assert sim.robot_model_key(0)=="autobot" and sim.robot_model_key(1)=="last_year"
    for bot in (0,1):
        cfg=sim.robot_config(bot)
        for yaw in (0.0,math.pi/2):
            sim.set_pose(bot,0.575,0.575,yaw)
            centre=sim.truth(bot)["position_m"][:2]
            wheels=np.asarray(sim.wheel_positions(bot))[:,:2]
            midpoint=wheels.mean(axis=0)
            rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            local=rot.T@(midpoint-centre)
            assert math.isclose(local[0],cfg["rear_axle_offset_m"],abs_tol=1e-6)
            assert local[0] < 0
            assert math.isclose(np.linalg.norm(wheels[0]-wheels[1]),cfg["wheel_spacing_m"],abs_tol=1e-6)

def test_seeded_reset_and_deterministic_trajectory():
    def run():
        sim=AutoBotSim(seed=4)
        return np.array([sim.step(tactic_to_drive(a))["position_m"] for a in [1,1,3,3,0]])
    assert np.array_equal(run(),run())

def test_tactic_directions_match_marked_chassis_front():
    forward=AutoBotSim(seed=4); start=forward.truth(0)
    for _ in range(10): forward.step(tactic_to_drive(1),(0,0))
    assert forward.truth(0)["position_m"][1] > start["position_m"][1]
    left=AutoBotSim(seed=4); start_yaw=left.truth(0)["yaw_rad"]
    for _ in range(10): left.step(tactic_to_drive(3),(0,0))
    yaw_delta=math.atan2(math.sin(left.truth(0)["yaw_rad"]-start_yaw),math.cos(left.truth(0)["yaw_rad"]-start_yaw))
    assert yaw_delta > 0

def test_versioned_forward_arcs_move_forward_and_yaw_as_named():
    results=[]
    for action in (5,6):
        env=AutoBotEnv(seed=44,action_profile="arc7_v1",opponent_mode="static"); env.reset(seed=44)
        start=env.sim.truth(0); start_pos=start["position_m"].copy(); start_yaw=start["yaw_rad"]
        for _ in range(25): env.step(action)
        end=env.sim.truth(0); yaw_delta=math.atan2(math.sin(end["yaw_rad"]-start_yaw),math.cos(end["yaw_rad"]-start_yaw))
        forward=np.array([math.cos(start_yaw),math.sin(start_yaw)])
        progress=float(np.dot(end["position_m"][:2]-start_pos[:2],forward)); results.append((progress,yaw_delta))
    assert results[0][0]>0.05 and results[0][1]>0.05
    assert results[1][0]>0.05 and results[1][1]<-0.05
    legacy=AutoBotEnv(seed=44,action_profile="legacy5")
    try: legacy.step(5)
    except ValueError: pass
    else: raise AssertionError("legacy profile accepted an arc action")

def test_manual_forward_reverse_match_marked_front_and_turns_stay_unchanged():
    forward=AutoBotSim(seed=8); start=forward.truth(0)["position_m"][1]
    for _ in range(10): forward.step(manual_drive_command("forward"),(0,0))
    assert forward.truth(0)["position_m"][1] > start
    reverse=AutoBotSim(seed=8); start=reverse.truth(0)["position_m"][1]
    for _ in range(10): reverse.step(manual_drive_command("reverse"),(0,0))
    assert reverse.truth(0)["position_m"][1] < start
    assert manual_drive_command("left")==(.7,-.7)
    assert manual_drive_command("right")==(-.7,.7)

def test_human_arcade_drive_supports_curves_and_neutral_release():
    assert arcade_drive()==(0.0,0.0)
    assert arcade_drive(forward=True)==(-1.0,-1.0)
    forward_left=arcade_drive(forward=True,left=True)
    assert forward_left[0]<0 and forward_left[1]<0 and forward_left[0]!=forward_left[1]
    assert arcade_drive(left=True)==(.7,-.7)

def test_top_wall_blocks_screenshot_escape_and_bot_can_drive_away():
    sim=AutoBotSim(seed=1); sim.set_pose(0,0.575,1.05,math.pi/2)
    for _ in range(100): sim.step((-1,-1),(0,0))
    assert sim.truth(0)["position_m"][1] < sim.cfg["arena"]["height_m"]
    assert sim.arena_event() is None
    wall_y=sim.truth(0)["position_m"][1]
    for _ in range(30): sim.step((1,1),(0,0))
    assert sim.truth(0)["position_m"][1] < wall_y-0.02

def test_both_bots_are_blocked_and_can_slide_along_top_wall():
    for bot in (0,1):
        sim=AutoBotSim(seed=bot+10); sim.set_pose(bot,0.30,1.08,math.pi/4)
        start_x=sim.truth(bot)["position_m"][0]
        for _ in range(80):
            commands=[(0,0),(0,0)]; commands[bot]=(-1,-1); sim.step(*commands)
        pose=sim.truth(bot)["position_m"]
        assert pose[1] < sim.cfg["arena"]["height_m"] and pose[0] > start_x+0.01
        assert sim.arena_event() is None

def test_all_four_wall_inner_faces_match_boundary_and_block_both_bots():
    reference=AutoBotSim(seed=30); arena=reference.cfg["arena"]
    for name,axis,expected in (("wall_left",0,0.0),("wall_right",0,arena["width_m"]),
                               ("wall_bottom",1,0.0),("wall_top",1,arena["height_m"])):
        geom=mujoco.mj_name2id(reference.model,mujoco.mjtObj.mjOBJ_GEOM,name)
        centre=reference.model.geom_pos[geom,axis]; half=reference.model.geom_size[geom,axis]
        inner=centre+half if expected==0 else centre-half
        assert math.isclose(inner,expected,abs_tol=1e-12)
    cases=((0.10,0.575,math.pi),(1.05,0.575,0.0),(0.575,0.10,-math.pi/2),(0.575,1.05,math.pi/2))
    for bot in (0,1):
        for x,y,yaw in cases:
            sim=AutoBotSim(seed=40+bot); sim.set_pose(bot,x,y,yaw)
            for _ in range(80):
                commands=[(0,0),(0,0)]; commands[bot]=(-1,-1); sim.step(*commands)
            px,py=sim.truth(bot)["position_m"][:2]
            assert 0 < px < arena["width_m"] and 0 < py < arena["height_m"]
            assert sim.arena_event() is None

def test_teleported_outer_wall_breach_is_diagnostic_not_match_loss():
    sim=AutoBotSim(seed=1); sim.set_pose(0,1.088,1.225,0.0)
    event=sim.arena_event()
    assert event.reason=="Simulation bug: outer-wall breach" and event.bot==0
    assert event.simulation_bug and not event.match_terminal

def test_own_pit_is_terminated_once_with_configured_reward_and_reset_clears_it():
    env=AutoBotEnv(seed=2); env.reset(seed=2)
    env.sim.set_pose(0,0.30,0.85,3*math.pi/4)
    for _ in range(100):
        _,reward,terminated,truncated,info=env.step(1)
        if terminated or truncated: break
    assert terminated and not truncated and reward == -1.0
    assert info["round_end_reason"]=="Pit fall" and info["round_end_bot"]==0
    try: env.step(0)
    except RuntimeError: pass
    else: raise AssertionError("terminal pit reward could be applied more than once")
    env.reset(seed=2)
    assert env.sim.arena_event() is None and not env.episode_done

def test_opponent_fall_is_a_configured_win():
    env=AutoBotEnv(seed=12,opponent_mode="static"); env.reset(seed=12)
    qadr=env.sim.model.jnt_qposadr[mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,"bot1_free")]
    env.sim.data.qpos[qadr:qadr+3]=[0.10,1.05,-0.02]
    mujoco.mj_forward(env.sim.model,env.sim.data)
    _,reward,terminated,truncated,info=env.step(0)
    assert terminated and not truncated and reward==1.0
    assert info["round_end_reason"]=="Pit fall" and info["round_end_bot"]==1
    assert info["reward_terms"]["opponent_fall"]==1.0

def test_simultaneous_fall_is_explicit_configured_draw():
    env=AutoBotEnv(seed=13,opponent_mode="static"); env.reset(seed=13)
    for bot in (0,1):
        qadr=env.sim.model.jnt_qposadr[mujoco.mj_name2id(env.sim.model,mujoco.mjtObj.mjOBJ_JOINT,f"bot{bot}_free")]
        env.sim.data.qpos[qadr+2]=-0.02
    mujoco.mj_forward(env.sim.model,env.sim.data)
    _,reward,terminated,truncated,info=env.step(0)
    assert terminated and not truncated and reward==0.0
    assert info["round_end_reason"]=="Simultaneous pit fall" and info["round_end_bot"] is None
    assert info["reward_terms"]["simultaneous_fall"]==0.0

def test_outcome_only_reward_has_no_movement_stop_distance_or_wall_term():
    for action in (0,1,2,3,4):
        env=AutoBotEnv(seed=14,opponent_mode="static"); env.reset(seed=14)
        _,reward,terminated,truncated,info=env.step(action)
        assert reward==0.0 and not terminated and not truncated
        assert sum(info["reward_terms"].values())==0.0
    env=AutoBotEnv(seed=15,opponent_mode="static"); env.reset(seed=15)
    env.sim.set_pose(0,0.575,0.07,-math.pi/2)
    contacted=False
    for _ in range(40):
        _,reward,terminated,truncated,info=env.step(1)
        contacted |= info["wall_contact"]
        assert reward==0.0 and not terminated
    assert contacted

def test_same_arena_rules_apply_to_opponent_and_rims_do_not_collide():
    cases=((0.30,0.85,3*math.pi/4),(0.85,0.30,-math.pi/4))
    for bot in (0,1):
        for x,y,yaw in cases:
            sim=AutoBotSim(seed=3+bot); sim.set_pose(bot,x,y,yaw)
            for _ in range(100):
                commands=[(0,0),(0,0)]; commands[bot]=(-1,-1); sim.step(*commands)
                if sim.arena_event(): break
            assert (sim.arena_event().reason,sim.arena_event().bot)==("Pit fall",bot)
    sim=AutoBotSim(seed=3)
    rim_ids=[i for i in range(sim.model.ngeom) if (mujoco.mj_id2name(sim.model,mujoco.mjtObj.mjOBJ_GEOM,i) or "").startswith("pit_rim_")]
    assert rim_ids and all(sim.model.geom_contype[i]==0 and sim.model.geom_conaffinity[i]==0 for i in rim_ids)

def test_verified_advance_positive_controls_and_stop_controls_at_both_pits():
    for pit in ("top_left","bottom_right"):
        centre=np.array([0.0,1.15]) if pit=="top_left" else np.array([1.15,0.0])
        inward=np.array([1.0,-1.0])/math.sqrt(2) if pit=="top_left" else np.array([-1.0,1.0])/math.sqrt(2)
        yaw=math.atan2(-inward[1],-inward[0]); red=centre+inward*0.32; blue=red+inward*0.18
        scenario={"name":pit,"player":[*blue,yaw],"opponent":[*red,yaw]}
        for mode in ("static","scripted"):
            stopped=AutoBotEnv(seed=31,opponent_mode=mode,opponent_model="last_year")
            stopped.reset(seed=31,options={"scenario":scenario})
            assert stopped.sim.truth(0)["position_m"][2]>0 and stopped.sim.truth(1)["position_m"][2]>0
            initial_gap=np.linalg.norm(stopped.sim.truth(0)["position_m"][:2]-stopped.sim.truth(1)["position_m"][:2])
            assert initial_gap>0.15 and not _bots_touch(stopped.sim)
            for _ in range(100):
                _,_,terminated,truncated,_=stopped.step(0)
                assert not terminated and not truncated
            advancing=AutoBotEnv(seed=31,opponent_mode=mode,opponent_model="last_year")
            advancing.reset(seed=31,options={"scenario":scenario}); red_start=advancing.sim.truth(1)["position_m"].copy(); touched=False
            for _ in range(100):
                _,reward,terminated,truncated,info=advancing.step(1); touched |= _bots_touch(advancing.sim)
                if terminated or truncated: break
            assert touched and np.linalg.norm(advancing.sim.truth(1)["position_m"][:2]-red_start[:2])>0.05
            assert terminated and not truncated and reward==1.0
            assert info["round_end_reason"]=="Pit fall" and info["round_end_bot"]==1

def test_partial_overhang_remains_supported_and_timeout_is_not_loss():
    sim=AutoBotSim(seed=20); centre=np.array(sim.cfg["arena"]["pit_centres_m"][0])
    supported=centre+np.array([1,-1])/math.sqrt(2)*0.255
    sim.set_pose(0,*supported,-math.pi/4)
    for _ in range(100): sim.step((0,0),(0,0))
    assert sim.truth(0)["position_m"][2] > 0 and sim.arena_event() is None
    env=AutoBotEnv(seed=21); env.reset(seed=21); env.max_steps=1
    _,_,terminated,truncated,info=env.step(0)
    assert not terminated and truncated and info["round_end_reason"]=="Timeout"
    assert info["reward_terms"]["timeout"]==0.0

def test_evaluation_scenario_override_and_default_spawn_regression():
    env=AutoBotEnv(seed=22)
    env.reset(seed=22)
    assert np.allclose(env.sim.truth(0)["position_m"][:2],[0.575,0.180])
    assert np.allclose(env.sim.truth(1)["position_m"][:2],[0.575,0.970])
    scenario={"name":"offset","player":[0.4,0.3,0.2],"opponent":[0.7,0.8,-0.4]}
    _,info=env.reset(seed=22,options={"scenario":scenario})
    assert info["scenario"]=="offset"
    assert np.allclose(env.sim.truth(0)["position_m"][:2],[0.4,0.3])
    assert np.allclose(env.sim.truth(1)["position_m"][:2],[0.7,0.8])

def test_scripted_opponent_stays_until_stepped_advances_and_resets():
    sim=AutoBotSim(seed=5); spawn=sim.truth(1)["position_m"][:2].copy()
    # Viewer pause is read-only: without a simulation step nothing can move.
    assert np.array_equal(sim.truth(1)["position_m"][:2],spawn)
    for _ in range(20): sim.step((0,0),scripted_opponent_drive(sim))
    assert sim.truth(1)["position_m"][1] < spawn[1]-0.05
    sim.reset()
    assert np.allclose(sim.truth(1)["position_m"][:2],spawn)

def test_static_opponent_drive_interface_does_not_move_it():
    sim=AutoBotSim(seed=6); spawn=sim.truth(1)["position_m"][:2].copy()
    for _ in range(20): sim.step((0,0),(0,0))
    assert np.linalg.norm(sim.truth(1)["position_m"][:2]-spawn) < 0.005

def test_gym_opponent_mode_is_explicit_and_does_not_change_observation_shape():
    scripted=AutoBotEnv(seed=61,opponent_mode="scripted",opponent_model="last_year")
    static=AutoBotEnv(seed=61,opponent_mode="static",opponent_model="last_year")
    obs_s,_=scripted.reset(seed=61); obs_t,_=static.reset(seed=61)
    assert obs_s.shape==obs_t.shape==(11,)
    for _ in range(10): scripted.step(0); static.step(0)
    assert scripted.sim.truth(1)["position_m"][1] < static.sim.truth(1)["position_m"][1]-0.005

def test_original_script_is_preserved_and_diagnostic_opponents_stay_in_range():
    sim=AutoBotSim(seed=62)
    assert opponent_drive(sim,"scripted")==scripted_opponent_drive(sim)
    for mode in ("static","faster_follow","stop_and_go","direction_changes"):
        command=opponent_drive(sim,mode)
        assert len(command)==2 and all(-1<=value<=1 for value in command)

def test_recorded_red_commands_replay_deterministically_and_reset_clears_round():
    commands=[(0.0,0.0),(-1.0,-.4),(-.4,-1.0),(0.65,0.65)]*8
    def run():
        env=AutoBotEnv(seed=63,opponent_mode="static"); env.reset(seed=63); trace=[]
        for command in commands:
            _,_,term,trunc,info=env.step(0,command)
            trace.append(np.r_[env.sim.truth(0)["position_m"],env.sim.truth(1)["position_m"]])
            assert tuple(info["opponent_drive"])==command
            assert not term and not trunc
        env.reset(seed=63)
        assert not env.episode_done and env.sim.data.time==0
        return np.asarray(trace)
    assert np.array_equal(run(),run())

def test_scripted_opponent_yields_at_contact_so_player_can_turn():
    sim=AutoBotSim(seed=7)
    for _ in range(60): sim.step(tactic_to_drive(1),scripted_opponent_drive(sim))
    assert scripted_opponent_drive(sim)==(0.0,0.0)
    start_yaw=sim.truth(0)["yaw_rad"]
    for _ in range(30): sim.step(tactic_to_drive(3),scripted_opponent_drive(sim))
    yaw_delta=math.atan2(math.sin(sim.truth(0)["yaw_rad"]-start_yaw),math.cos(sim.truth(0)["yaw_rad"]-start_yaw))
    assert yaw_delta > 0.2
