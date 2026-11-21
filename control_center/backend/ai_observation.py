"""Is the AI's observation complete and fresh? Pure functions: no Qt, no clock, no camera, no port, nothing sent.

Input: whatever the shadow/tracking code already produced, in any of these shapes (nothing here imports them):

  * a ``stream_adapter.Decision`` (has ``inputs``: group -> InputStatus with ``valid``, ``age_ms``, ``reason``, ``source``);
  * the dictionary ``ShadowRunner.decide`` returns (``inputs``: group -> {valid, reason, source}); it carries NO ages;
  * a ``tracking_worker.TrackingResult`` (has ``shadow`` dict, ``own`` and ``opponent`` track dictionaries);
  * ``Decision.to_trace()`` (``inputs`` with ``age_ms``).

Output: ``ObservationStatus``: per group valid / age / reason, complete, stale, what is missing, and one plain sentence.

Rules, none of them negotiable:
  * a value that was not measured is MISSING; nothing is filled in, held, or taken from the simulator;
  * the IMU is real data or it is missing: when no IMU line has arrived the reason is IMU_ABSENT_REASON;
  * a group is fresh only when its age is KNOWN and within its limit (camera 250 ms, IMU 100 ms). An input that
    carries no age (the ShadowRunner dictionary) is shown as valid-as-reported, with ``age_ms`` None and ``age_known``
    False, and is NOT counted as fresh by the autonomous gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

GROUPS = ("imu", "previous", "own_pose", "opponent")
# The observation contract's own group names (control_center.tools.observation_contract) -> the short names used here.
CONTRACT_NAMES = {"imu": "imu", "previous_action": "previous", "own_pose": "own_pose", "opponent_position": "opponent"}
CAMERA_MAX_AGE_MS = 250.0
IMU_MAX_AGE_MS = 100.0
IMU_ABSENT_REASON = "the baseline firmware sends no IMU data: nothing is invented"
OBSERVATION_VALUES = 19
OWN_VALID_STATES = ("measured", "predicted")
LABELS = {"imu": "IMU", "previous": "previous tactic", "own_pose": "own pose", "opponent": "opponent"}


@dataclass(frozen=True)
class GroupStatus:
    group: str
    valid: bool                    # present, and fresh when its age is known
    age_ms: float | None
    reason: str                    # "ok" or exactly why not
    source: str = ""
    stale: bool = False
    age_known: bool = False


@dataclass(frozen=True)
class ObservationStatus:
    groups: dict[str, GroupStatus]
    complete: bool                 # every group valid, none stale, and (when known) the vector was assembled
    stale: bool                    # at least one input is older than its limit
    missing: list[str]             # "own pose: <reason>" for each group that is not valid
    statement: str                 # the plain sentence for the operator
    own_state: str | None = None
    opponent_state: str | None = None
    tracking_ok: bool = False      # own pose measured/predicted AND opponent tracked
    tracking_reason: str = ""
    observation_size: int | None = None
    fresh_known: bool = False      # True only when every ageing group has a known age within its limit
    notes: list[str] = field(default_factory=list)

    def oldest_input_age_ms(self) -> float | None:
        ages = [g.age_ms for g in self.groups.values() if g.age_ms is not None and g.valid]
        return max(ages) if ages else None


def _get(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, Mapping):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _unwrap(source: Any) -> tuple[Mapping[str, Any], Any, Any, Any, str]:
    """(inputs, own track, opponent track, assembled observation or marker, error text)."""
    if source is None:
        return {}, None, None, None, ""
    shadow = _get(source, "shadow")
    if shadow is not None or (not isinstance(source, Mapping) and hasattr(source, "own") and hasattr(source, "opponent")):
        # A TrackingResult: the decision (if any) is in .shadow
        inner = shadow if isinstance(shadow, Mapping) else {}
        return (inner.get("inputs", {}) or {}, _get(source, "own"), _get(source, "opponent"), inner.get("observation"),
                str(inner.get("error", "") or ""))
    inputs = _get(source, "inputs", {}) or {}
    return inputs, None, None, _get(source, "observation"), str(_get(source, "error", "") or "")


def _group_entry(inputs: Mapping[str, Any], short: str) -> Any:
    for contract, name in CONTRACT_NAMES.items():
        if name == short:
            if contract in inputs:
                return inputs[contract]
            if short in inputs:
                return inputs[short]
    return None


def _evaluate(short: str, entry: Any) -> GroupStatus:
    if entry is None:
        if short == "previous":
            return GroupStatus(short, True, None, "not used by this checkpoint", "", False, False)
        reason = IMU_ABSENT_REASON if short == "imu" else "no input has been reported for this group"
        return GroupStatus(short, False, None, reason)
    valid = bool(_get(entry, "valid", False))
    age = _get(entry, "age_ms")
    age = None if age is None else float(age)
    reason = str(_get(entry, "reason", "") or ("ok" if valid else "not valid"))
    source = str(_get(entry, "source", "") or "")
    if short == "previous":
        return GroupStatus(short, valid, None, reason, source, False, False)
    limit = IMU_MAX_AGE_MS if short == "imu" else CAMERA_MAX_AGE_MS
    if short == "imu" and not valid and (source in ("", "none") or age is None) and not reason.lower().startswith("stale"):
        return GroupStatus(short, False, age, IMU_ABSENT_REASON, source, False, age is not None)
    stale = reason.lower().startswith("stale") or (age is not None and age > limit)
    if valid and stale:
        valid = False
        reason = reason if reason.lower().startswith("stale") else f"stale: {age:.0f} ms old (limit {limit:.0f} ms)"
    return GroupStatus(short, valid, age, reason, source, stale, age is not None)


def _state(track: Any) -> str | None:
    value = _get(track, "state")
    if value is None:
        return None
    return str(getattr(value, "value", value)).lower()


def observation_status(source: Any, *, elapsed_s: float = 0.0) -> ObservationStatus:
    """``elapsed_s``: how long ago the decision was made, added to every known age (a decision ages while nobody looks)."""
    inputs, own, opponent, assembled, error = _unwrap(source)
    shifted: dict[str, GroupStatus] = {}
    for short in GROUPS:
        entry = _group_entry(inputs, short)
        status = _evaluate(short, entry)
        if elapsed_s > 0 and status.age_ms is not None and short != "previous":
            entry = {"valid": bool(_get(entry, "valid", False)), "age_ms": status.age_ms + 1000.0 * elapsed_s,
                     "reason": _get(entry, "reason", ""), "source": status.source}
            status = _evaluate(short, entry)
        shifted[short] = status
    missing = [f"{LABELS[g]}: {s.reason}" for g, s in shifted.items() if not s.valid]
    stale = any(s.stale for s in shifted.values())
    notes: list[str] = []
    size = None
    if assembled is not None and not isinstance(assembled, str):
        try:
            size = len(assembled)
        except TypeError:
            size = None
    vector_ok = True
    if source is not None and (_get(source, "observation", "absent") is None) and not missing:
        vector_ok = False
        notes.append("every group reports valid but the 19-value vector was not assembled")
    complete = bool(inputs) and not missing and not stale and vector_ok
    ageing = [shifted[g] for g in ("imu", "own_pose", "opponent")]
    fresh_known = all(s.valid and s.age_known for s in ageing)
    if not fresh_known and complete:
        notes.append("input ages were not reported: freshness cannot be confirmed from this data")
    own_state, opponent_state = _state(own), _state(opponent)
    tracking_ok, tracking_reason = False, ""
    own_ok = (own_state in OWN_VALID_STATES) if own_state is not None else shifted["own_pose"].valid
    opp_ok = (opponent_state == "tracked") if opponent_state is not None else shifted["opponent"].valid
    if not inputs and own is None and opponent is None:
        tracking_reason = "no tracking result or decision yet"
    elif not own_ok:
        tracking_reason = f"own pose is not valid ({own_state or shifted['own_pose'].reason})"
    elif not opp_ok:
        tracking_reason = f"the opponent is not tracked ({opponent_state or shifted['opponent'].reason})"
    else:
        tracking_ok, tracking_reason = True, "own pose measured/predicted and the opponent is tracked"
    if not inputs:
        statement = "No observation yet: no decision has been made, so nothing can be said about the inputs."
    elif complete:
        statement = (f"Observation COMPLETE: all four input groups are valid and none is stale"
                     + (f" ({size} values assembled)." if size else "."))
        if not fresh_known:
            statement += " Input ages were not reported, so freshness is NOT confirmed."
    else:
        parts = "; ".join(missing) if missing else "; ".join(notes)
        statement = ("Observation INCOMPLETE" + (" and STALE" if stale else "") + f": {parts}. Nothing is invented to fill the gap.")
    if error:
        statement += f" Policy error: {error}."
    return ObservationStatus(shifted, complete, stale, missing, statement, own_state, opponent_state, tracking_ok,
                             tracking_reason, size, fresh_known, notes)
