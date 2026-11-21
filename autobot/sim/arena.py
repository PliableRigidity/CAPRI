"""Shared arena geometry and MuJoCo-state event classification."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ArenaEvent:
    reason: str
    bot: int | None
    match_terminal: bool = True
    simulation_bug: bool = False


def playable_floor_polygon(arena: dict) -> list[np.ndarray]:
    """Clockwise outer floor perimeter with the two R250 corner cutouts."""
    w,h,r=arena["width_m"],arena["height_m"],arena["pit_radius_m"]
    count=int(arena.get("pit_arc_segments",20))
    points=[np.array([0.0,0.0]),np.array([0.0,h-r])]
    points += [np.array([r*math.cos(a),h+r*math.sin(a)]) for a in np.linspace(-math.pi/2,0,count+1)[1:]]
    points += [np.array([w,h]),np.array([w,r])]
    points += [np.array([w+r*math.cos(a),r*math.sin(a)]) for a in np.linspace(math.pi/2,math.pi,count+1)[1:]]
    return points


def _cross(a,b,c):
    ab=b-a; ac=c-a
    return float(ab[0]*ac[1]-ab[1]*ac[0])


def triangulate_polygon(points: list[np.ndarray]) -> list[tuple[np.ndarray,np.ndarray,np.ndarray]]:
    """Ear-clip a simple polygon into convex triangles for collision prisms."""
    vertices=[np.asarray(p,dtype=float) for p in points]
    signed=sum(float(vertices[i][0]*vertices[(i+1)%len(vertices)][1]-vertices[i][1]*vertices[(i+1)%len(vertices)][0]) for i in range(len(vertices)))/2
    if signed < 0: vertices.reverse()
    indices=list(range(len(vertices))); triangles=[]
    while len(indices)>3:
        clipped=False
        for slot,index in enumerate(indices):
            prev=indices[slot-1]; nxt=indices[(slot+1)%len(indices)]
            a,b,c=vertices[prev],vertices[index],vertices[nxt]
            if _cross(a,b,c) <= 1e-12: continue
            def inside(p):
                return _cross(a,b,p)>=-1e-12 and _cross(b,c,p)>=-1e-12 and _cross(c,a,p)>=-1e-12
            if any(inside(vertices[j]) for j in indices if j not in (prev,index,nxt)): continue
            triangles.append((a,b,c)); indices.pop(slot); clipped=True; break
        if not clipped: raise ValueError("Could not triangulate arena floor polygon")
    triangles.append(tuple(vertices[i] for i in indices))
    return triangles


def floor_triangles(arena: dict):
    return triangulate_polygon(playable_floor_polygon(arena))


def classify_sim_state(positions_xyz, arena: dict) -> ArenaEvent | None:
    """Fall is a match result; passing a wall is a diagnostic bug, not a loss."""
    fallen=[bot for bot,position in enumerate(positions_xyz)
            if float(position[2]) < arena["fall_height_m"]]
    # Provisional simulator convention: a fall detected for both bots in the same
    # control step is a draw. This is explicit, configurable, and not an official rule.
    if len(fallen)==2:
        return ArenaEvent("Simultaneous pit fall",None)
    if fallen:
        return ArenaEvent("Pit fall",fallen[0])
    margin=float(arena["wall_breach_margin_m"]); w=arena["width_m"]; h=arena["height_m"]
    for bot,position in enumerate(positions_xyz):
        x,y=float(position[0]),float(position[1])
        if x < -margin or x > w+margin or y < -margin or y > h+margin:
            return ArenaEvent("Simulation bug: outer-wall breach",bot,match_terminal=False,simulation_bug=True)
    return None


def pit_rim_segments(arena: dict, segment_count: int | None = None):
    """Line segments for the two configured in-arena quarter-circle arcs."""
    segment_count=segment_count or int(arena.get("pit_arc_segments",20)); result=[]
    for centre in arena["pit_centres_m"]:
        cx,cy=centre
        if math.isclose(cx,0.0) and math.isclose(cy,arena["height_m"]): angles=np.linspace(-math.pi/2,0,segment_count+1)
        elif math.isclose(cx,arena["width_m"]) and math.isclose(cy,0.0): angles=np.linspace(math.pi/2,math.pi,segment_count+1)
        else: continue
        points=[np.array([cx+arena["pit_radius_m"]*math.cos(a),cy+arena["pit_radius_m"]*math.sin(a)]) for a in angles]
        result.extend(zip(points[:-1],points[1:]))
    return result
