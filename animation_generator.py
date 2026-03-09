import os
import re
import math
import numpy as np

# ---------------------------------------------------------------------------
# animation_generator.py  (v3 – full procedural skeleton animation)
#
# Pipeline:
#   1. Use sentence-transformers to classify the prompt into one of the
#      built-in action profiles (walk, run, jump, wave, dance, …).
#   2. Drive a 22-joint forward-kinematics skeleton with anatomically-
#      plausible, time-varying joint angles computed by a keyframe engine
#      based on scipy CubicSpline interpolation.
#   3. Render each frame to a PIL Image (glowing amber joints on dark bg).
#   4. Assemble frames with moviepy into an MP4 (up to 30 s at 30 fps).
# ---------------------------------------------------------------------------

VALID_FILENAME = re.compile(r"[^a-zA-Z0-9_-]")

# ---------------------------------------------------------------------------
# Module-level STS cache
# ---------------------------------------------------------------------------
_sts_model = None
_sts_corpus_embeddings = None
_sts_corpus_labels = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_hint(hint: str) -> str:
    cleaned = hint.strip().lower().replace(" ", "_")
    cleaned = VALID_FILENAME.sub("", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "animation"


# ---------------------------------------------------------------------------
# Action Classification
# ---------------------------------------------------------------------------

_ACTION_KEYWORDS = {
    "run":     ["run", "sprint", "jog", "race", "chase", "rush", "flee", "dash"],
    "walk":    ["walk", "stroll", "march", "hike", "pace", "wander", "step"],
    "jump":    ["jump", "leap", "hop", "bounce", "skip", "vault", "spring"],
    "wave":    ["wave", "hello", "hi ", "hey", "greet", "goodbye", "bye"],
    "dance":   ["dance", "party", "celebrate", "groove", "twirl", "rhythm"],
    "fight":   ["fight", "punch", "kick", "attack", "battle", "combat", "strike", "hit", "defend"],
    "fall":    ["fall", "trip", "slip", "tumble", "collapse", "drop", "faint"],
    "reach":   ["reach", "grab", "extend", "take", "lift up"],
    "turn":    ["turn", "rotate", "look around", "look back", "pivot"],
    "sit":     ["sit", "seat", "couch", "chair", "bench", "kneel", "crouch"],
    "climb":   ["climb", "crawl", "scale", "ascend"],
    "swim":    ["swim", "float", "dive", "splash"],
    "sneak":   ["sneak", "creep", "tiptoe", "stealth"],
    "throw":   ["throw", "toss", "fling", "pitch", "hurl"],
    # ── Daily-life actions ────────────────────────────────────────────────
    "eat":     ["eat", "eating", "food", "chew", "bite", "meal", "lunch", "dinner", "breakfast", "snack"],
    "sleep":   ["sleep", "sleeping", "nap", "lie down", "rest", "snore", "slumber", "doze", "bed"],
    "drink":   ["drink", "drinking", "sip", "gulp", "water", "coffee", "tea", "cup", "beverage"],
    "read":    ["read", "reading", "book", "newspaper", "magazine", "study", "look at"],
    "phone":   ["phone", "call", "talk on phone", "mobile", "cellphone", "text", "dial"],
    "bow":     ["bow", "bowing", "respect", "salute", "greet bow", "curtsy"],
    "think":   ["think", "thinking", "ponder", "wonder", "contemplate", "scratch head"],
    "stretch": ["stretch", "stretching", "yoga", "warm up", "reach up", "arms up"],
    "push":    ["push", "pushing", "shove", "press forward", "push door"],
    "pull":    ["pull", "pulling", "yank", "drag", "tug"],
    "idle":    [],  # fallback
}


def _classify_actions(prompt: str) -> list:
    """Return a list of (action, weight) pairs detected in the prompt.

    Supports combinations like "walk and wave", "run then jump", "eat while reading".
    Returns at most 2 actions to keep blending tractable.
    Weights are normalised so they sum to 1.0.
    """
    # Split on common combination connectors
    _COMBO_SEPS = [" and ", " while ", " then ", " + ", " & ", " with ", " then stop and "]
    low = prompt.lower()
    # Try to split the prompt into sub-prompts
    sub_prompts = [low]
    for sep in _COMBO_SEPS:
        if sep in low:
            parts = low.split(sep, 1)
            sub_prompts = [p.strip() for p in parts if p.strip()]
            break

    found = []
    for sub in sub_prompts[:2]:  # cap at 2
        # Keyword scan on sub-prompt
        matched = None
        for action, keywords in _ACTION_KEYWORDS.items():
            if action == "idle":
                continue
            for kw in keywords:
                if kw in sub:
                    matched = action
                    break
            if matched:
                break
        if matched and matched not in [a for a, _ in found]:
            found.append((matched, 1.0))

    # Semantic fallback if nothing found
    if not found:
        try:
            global _sts_model, _sts_corpus_embeddings, _sts_corpus_labels
            from sentence_transformers import SentenceTransformer, util
            labels = [a for a in _ACTION_KEYWORDS if a != "idle"]
            if _sts_model is None:
                model_path = os.path.join(os.path.dirname(__file__), "model", "finetuned_model")
                _sts_model = SentenceTransformer(model_path) if os.path.exists(model_path) else SentenceTransformer("all-MiniLM-L6-v2")
            if _sts_corpus_labels != labels:
                _sts_corpus_labels = labels
                _sts_corpus_embeddings = _sts_model.encode(labels, convert_to_tensor=True)
            prompt_emb = _sts_model.encode(prompt, convert_to_tensor=True)
            scores = util.cos_sim(prompt_emb, _sts_corpus_embeddings)[0]
            best = labels[int(scores.argmax())]
            found = [(best, 1.0)]
        except Exception:
            found = [("idle", 1.0)]

    # Normalise weights (if two actions, blend 50/50 unless one is locomotion)
    if len(found) == 2:
        loco = {"walk", "run", "sneak", "climb", "swim"}
        a0, a1 = found[0][0], found[1][0]
        # Give locomotion 60% weight when paired with another action
        if a0 in loco and a1 not in loco:
            found = [(a0, 0.6), (a1, 0.4)]
        elif a1 in loco and a0 not in loco:
            found = [(a0, 0.4), (a1, 0.6)]
        else:
            found = [(a0, 0.5), (a1, 0.5)]
    return found


def _classify_action(prompt: str) -> str:
    """Backward-compatible single-action classifier."""
    return _classify_actions(prompt)[0][0]


def _blend_poses(pose_a: dict, pose_b: dict, weight_a: float) -> dict:
    """Linearly blend two pose dicts. weight_a in [0, 1]."""
    wb = 1.0 - weight_a
    result = {}
    all_keys = set(pose_a) | set(pose_b)
    for k in all_keys:
        va = pose_a.get(k, 0.0)
        vb = pose_b.get(k, 0.0)
        result[k] = va * weight_a + vb * wb
    return result


# ---------------------------------------------------------------------------
# Skeleton Definition
# ---------------------------------------------------------------------------
# Joint index map
J = {
    "hips": 0, "spine": 1, "spine1": 2, "spine2": 3, "neck": 4, "head": 5,
    "l_shoulder": 6, "l_elbow": 7, "l_wrist": 8,
    "r_shoulder": 9, "r_elbow": 10, "r_wrist": 11,
    "l_hip": 12, "l_knee": 13, "l_ankle": 14, "l_toe": 15,
    "r_hip": 16, "r_knee": 17, "r_ankle": 18, "r_toe": 19,
}

# Bone lengths (relative units, normalised to 100px spine height)
# ---------------------------------------------------------------------------
# Human-accurate skeleton proportions
# ---------------------------------------------------------------------------
# All lengths are fractions of total body height.
# Reference: average adult standing height = 1.0 unit.
# Head      = 1/8  (0.125)
# Neck      = 0.06
# Spine (3) = 0.32 total
# Upper arm = 0.186, Forearm = 0.146, Shoulder offset = 0.23 of total
# Thigh     = 0.245, Shin = 0.245, Foot = 0.075
_BONE_LEN: dict = {
    ("hips",  "spine"):       0.095,   # lumbar
    ("spine",  "spine1"):     0.105,   # thoracic lower
    ("spine1", "spine2"):     0.11,    # thoracic upper / shoulder level
    ("spine2", "neck"):       0.07,    # lower neck
    ("neck",   "head"):       0.125,   # head (circle rendered separately)
    # Arms (shoulder is an offset, not bone length)
    ("spine2", "l_shoulder"): 0.23,    # shoulder half-width offset
    ("spine2", "r_shoulder"): 0.23,
    ("l_shoulder", "l_elbow"): 0.186,  # humerus
    ("l_elbow",   "l_wrist"): 0.146,   # radius
    ("r_shoulder", "r_elbow"): 0.186,
    ("r_elbow",   "r_wrist"): 0.146,
    # Legs (hip is a half-width offset, not bone length)
    ("hips", "l_hip"):        0.105,   # hip half-width offset
    ("l_hip",  "l_knee"):     0.245,   # femur
    ("l_knee", "l_ankle"):    0.245,   # tibia
    ("l_ankle", "l_toe"):     0.075,   # foot
    ("hips", "r_hip"):        0.105,
    ("r_hip",  "r_knee"):     0.245,
    ("r_knee", "r_ankle"):    0.245,
    ("r_ankle", "r_toe"):     0.075,
}

# Skeleton pixel height (total body height in pixels)
_SKEL_SCALE = 360


def _bone_px(name: str) -> int:
    for (p, c), v in _BONE_LEN.items():
        if c == name:
            return max(4, int(v * _SKEL_SCALE))
    return max(4, int(0.10 * _SKEL_SCALE))

# (parent, child) connectivity used for rendering
SKELETON_BONES = [
    ("hips", "spine"), ("spine", "spine1"), ("spine1", "spine2"),
    ("spine2", "neck"), ("neck", "head"),
    ("spine2", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("spine2", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("hips", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_toe"),
    ("hips", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe"),
]

# Scale: skeleton height in pixels
_SKEL_SCALE = 280


def _bone_px(name: str) -> int:
    for (p, c), v in _BONE_LEN.items():
        if c == name:
            return int(v * _SKEL_SCALE)
    return int(0.15 * _SKEL_SCALE)


# ---------------------------------------------------------------------------
# Keyframe Engine
# ---------------------------------------------------------------------------

def _interp(t: float, keyframes: list) -> float:
    """Linearly interpolate between (t_frac, value) keyframes, clamped."""
    if not keyframes:
        return 0.0
    if t <= keyframes[0][0]:
        return keyframes[0][1]
    if t >= keyframes[-1][0]:
        return keyframes[-1][1]
    for i in range(len(keyframes) - 1):
        t0, v0 = keyframes[i]
        t1, v1 = keyframes[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0 + 1e-9)
            # Smooth-step
            alpha = alpha * alpha * (3 - 2 * alpha)
            return v0 + alpha * (v1 - v0)
    return keyframes[-1][1]


def _sine(t: float, freq: float, amp: float, phase: float = 0.0) -> float:
    return amp * math.sin(2 * math.pi * freq * t + phase)


# ---------------------------------------------------------------------------
# Action profiles — return joint angles in degrees for time t (in seconds)
# ---------------------------------------------------------------------------

def _profile_idle(t: float) -> dict:
    breathe  = _sine(t, 0.22, 1.2)          # slow chest rise
    head_nod = _sine(t, 0.18, 1.5)          # very slow head bob
    sway     = _sine(t, 0.12, 0.5)          # micro body sway
    return {
        "spine_bend":   1.5 + breathe * 0.4,
        "head_pitch":   head_nod,
        "head_yaw":     sway * 3.0,
        "l_shoulder_x": -8 + breathe * 0.3,
        "r_shoulder_x": -8 + breathe * 0.3,
        "l_elbow": 12,
        "r_elbow": 12,
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee":  3,  "r_knee":  3,
        "l_ankle": 0,  "r_ankle": 0,
        "hip_sway":     sway * 2.0,
        "hip_roll":     sway * 1.5,
        "hip_y_offset": 0,
    }


def _profile_walk(t: float) -> dict:
    # Stride rate: ~1.8 Hz matches brisk walking cadence
    cycle  = t * 1.8
    stride = math.sin(2 * math.pi * cycle)          # [-1, 1] leg phase
    arm    = -stride                                 # contralateral arm swing
    # Double-support dip: hips drop slightly twice per stride cycle
    hip_dip = abs(math.sin(2 * math.pi * cycle * 2)) * -4
    return {
        # Slight forward lean — typical 5-7 deg for brisk walk
        "spine_bend":   5 + _sine(t, cycle * 2, 0.5),
        "head_pitch":   3,
        "head_yaw":     stride * 3,
        # Arms swing fore/aft with elbow slightly flexed
        "l_shoulder_x": arm * 32,
        "r_shoulder_x": -arm * 32,
        "l_elbow":      30 + arm * 8,
        "r_elbow":      30 - arm * 8,
        # Contralateral leg swing  (+fwd = flexion, -back = extension)
        "l_hip_x":  stride * 32,
        "r_hip_x": -stride * 32,
        # Knee bends mainly on the swing leg; stance leg stays straighter
        "l_knee":  max(0, -stride * 42) + 4,
        "r_knee":  max(0, stride * 42) + 4,
        # Ankle: push off on trailing foot, dorsiflex on leading foot
        "l_ankle":  stride * 18,
        "r_ankle": -stride * 18,
        # Pelvis shifts laterally over support leg & rolls with stride
        "hip_sway":     stride * 5,
        "hip_roll":     stride * 4,
        "hip_y_offset": hip_dip,
    }


def _profile_run(t: float) -> dict:
    # Stride rate: ~2.8 Hz for a jog, increased hip/trunk drive
    cycle  = t * 2.8
    stride = math.sin(2 * math.pi * cycle)
    arm    = -stride
    # Air-time: both feet off ground, hips rise slightly at mid-flight
    flight = abs(math.sin(2 * math.pi * cycle * 2)) * -10
    return {
        # Forward lean typical 10-15 deg for running
        "spine_bend":   12 + _sine(t, cycle * 2, 1.2),
        "head_pitch":   5,
        "head_yaw":     stride * 4,
        # Vigorous arm drive with high elbow flexion (~90 deg)
        "l_shoulder_x": arm * 58,
        "r_shoulder_x": -arm * 58,
        "l_elbow":      85 + arm * 12,
        "r_elbow":      85 - arm * 12,
        # High knee drive + strong hip extension on opposite leg
        "l_hip_x":   stride * 58,
        "r_hip_x":  -stride * 58,
        "l_knee":   max(0, -stride * 85) + 6,
        "r_knee":   max(0, stride * 85) + 6,
        # Ankle plantarflexion on push-off, dorsiflexion on swing
        "l_ankle":  stride * 28,
        "r_ankle": -stride * 28,
        "hip_sway":     stride * 7,
        "hip_roll":     stride * 6,
        "hip_y_offset": flight,
    }


def _profile_jump(t: float, duration: float) -> dict:
    # Phases: 0-20% crouch, 20-40% ascent, 40-60% apex, 60-80% descent, 80-100% land
    norm = t / duration
    if norm < 0.20:   # crouch
        bend = _interp(norm / 0.20, [(0, 0), (1, 45)])
        return {"spine_bend": 10, "head_pitch": 10, "head_yaw": 0,
                "l_shoulder_x": -20, "r_shoulder_x": -20,
                "l_elbow": 40, "r_elbow": 40,
                "l_hip_x": bend, "r_hip_x": bend,
                "l_knee": bend * 1.8, "r_knee": bend * 1.8,
                "l_ankle": -20, "r_ankle": -20,
                "hip_sway": 0, "hip_y_offset": _interp(norm / 0.20, [(0, 0), (1, 24)])}
    elif norm < 0.40: # ascent
        p = (norm - 0.20) / 0.20
        return {"spine_bend": 5, "head_pitch": -10, "head_yaw": 0,
                "l_shoulder_x": _interp(p, [(0, -20), (1, -90)]),
                "r_shoulder_x": _interp(p, [(0, -20), (1, -90)]),
                "l_elbow": _interp(p, [(0, 40), (1, 10)]),
                "r_elbow": _interp(p, [(0, 40), (1, 10)]),
                "l_hip_x": _interp(p, [(0, 45), (1, -20)]),
                "r_hip_x": _interp(p, [(0, 45), (1, -20)]),
                "l_knee": _interp(p, [(0, 80), (1, 10)]),
                "r_knee": _interp(p, [(0, 80), (1, 10)]),
                "l_ankle": _interp(p, [(0, -20), (1, 30)]),
                "r_ankle": _interp(p, [(0, -20), (1, 30)]),
                "hip_sway": 0,
                "hip_y_offset": _interp(p, [(0, 24), (1, -70)])}
    elif norm < 0.60: # apex
        return {"spine_bend": 0, "head_pitch": -5, "head_yaw": 0,
                "l_shoulder_x": -90, "r_shoulder_x": -90,
                "l_elbow": 10, "r_elbow": 10,
                "l_hip_x": -15, "r_hip_x": -15,
                "l_knee": 5, "r_knee": 5,
                "l_ankle": 30, "r_ankle": 30,
                "hip_sway": 0, "hip_y_offset": -70}
    elif norm < 0.80: # descent
        p = (norm - 0.60) / 0.20
        return {"spine_bend": 5, "head_pitch": 5, "head_yaw": 0,
                "l_shoulder_x": _interp(p, [(0, -90), (1, -20)]),
                "r_shoulder_x": _interp(p, [(0, -90), (1, -20)]),
                "l_elbow": _interp(p, [(0, 10), (1, 40)]),
                "r_elbow": _interp(p, [(0, 10), (1, 40)]),
                "l_hip_x": _interp(p, [(0, -15), (1, 30)]),
                "r_hip_x": _interp(p, [(0, -15), (1, 30)]),
                "l_knee": _interp(p, [(0, 5), (1, 60)]),
                "r_knee": _interp(p, [(0, 5), (1, 60)]),
                "l_ankle": _interp(p, [(0, 30), (1, -10)]),
                "r_ankle": _interp(p, [(0, 30), (1, -10)]),
                "hip_sway": 0,
                "hip_y_offset": _interp(p, [(0, -70), (1, 10)])}
    else:             # land + recover
        p = (norm - 0.80) / 0.20
        bend = _interp(p, [(0, 40), (1, 0)])
        return {"spine_bend": _interp(p, [(0, 15), (1, 3)]),
                "head_pitch": _interp(p, [(0, 10), (1, 0)]),
                "head_yaw": 0,
                "l_shoulder_x": _interp(p, [(0, -20), (1, -5)]),
                "r_shoulder_x": _interp(p, [(0, -20), (1, -5)]),
                "l_elbow": _interp(p, [(0, 40), (1, 10)]),
                "r_elbow": _interp(p, [(0, 40), (1, 10)]),
                "l_hip_x": bend, "r_hip_x": bend,
                "l_knee": bend * 1.5, "r_knee": bend * 1.5,
                "l_ankle": _interp(p, [(0, -15), (1, 0)]),
                "r_ankle": _interp(p, [(0, -15), (1, 0)]),
                "hip_sway": 0,
                "hip_y_offset": _interp(p, [(0, 10), (1, 0)])}


def _profile_wave(t: float) -> dict:
    wave_arm = _sine(t, 1.5, 40.0) + 80  # oscillates 40–120°
    wave_hand = _sine(t, 2.0, 25.0, phase=0.5)
    return {
        "spine_bend": 2,
        "head_pitch": 5,
        "head_yaw": 10,
        "l_shoulder_x": -10,
        "r_shoulder_x": -wave_arm,
        "l_elbow": 15,
        "r_elbow": max(0, 90 + wave_hand),
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 2, "r_knee": 2,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": _sine(t, 0.3, 2.5),
        "hip_y_offset": 0,
    }


def _profile_dance(t: float) -> dict:
    beat = math.sin(2 * math.pi * 1.5 * t)
    hip  = math.sin(2 * math.pi * 3.0 * t)
    arm1 = _sine(t, 1.5, 55, 0)
    arm2 = _sine(t, 1.5, 55, math.pi)
    return {
        "spine_bend": 5 + beat * 4,
        "head_pitch": beat * 6,
        "head_yaw": hip * 12,
        "l_shoulder_x": arm1,
        "r_shoulder_x": arm2,
        "l_elbow": 50 + beat * 20,
        "r_elbow": 50 - beat * 20,
        "l_hip_x": hip * 20,
        "r_hip_x": -hip * 20,
        "l_knee": max(0, hip * 18),
        "r_knee": max(0, -hip * 18),
        "l_ankle": beat * 10, "r_ankle": -beat * 10,
        "hip_sway": hip * 14,
        "hip_y_offset": abs(beat) * -5,
    }


def _profile_fight(t: float) -> dict:
    cycle  = t * 2.5
    jab_l  = math.sin(2 * math.pi * cycle)
    jab_r  = math.sin(2 * math.pi * cycle + math.pi)
    weight = math.sin(2 * math.pi * cycle * 0.5)
    return {
        "spine_bend": 15,
        "head_pitch": 5,
        "head_yaw": weight * 15,
        "l_shoulder_x": 20 + jab_l * 60,
        "r_shoulder_x": 20 + jab_r * 60,
        "l_elbow": max(0, 80 - jab_l * 80),
        "r_elbow": max(0, 80 - jab_r * 80),
        "l_hip_x": weight * 18,
        "r_hip_x": -weight * 18,
        "l_knee": 20 + weight * 15,
        "r_knee": 20 - weight * 15,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": weight * 10,
        "hip_y_offset": 8,
    }


def _profile_fall(t: float, duration: float) -> dict:
    norm = t / duration
    rot  = min(1.0, norm * 1.5)  # fall completes at 66% of duration; then stays
    return {
        "spine_bend": rot * 80 + _sine(t, 5, 2) * max(0, 1 - norm * 2),
        "head_pitch": rot * 50,
        "head_yaw": rot * 20,
        "l_shoulder_x": _interp(rot, [(0, -5), (0.5, -60), (1, -30)]),
        "r_shoulder_x": _interp(rot, [(0, -5), (0.5, 80), (1, 60)]),
        "l_elbow": _interp(rot, [(0, 10), (1, 60)]),
        "r_elbow": _interp(rot, [(0, 10), (1, 50)]),
        "l_hip_x": _interp(rot, [(0, 0), (1, 30)]),
        "r_hip_x": _interp(rot, [(0, 0), (1, 20)]),
        "l_knee": _interp(rot, [(0, 2), (1, 35)]),
        "r_knee": _interp(rot, [(0, 2), (1, 20)]),
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": rot * 15,
        "hip_y_offset": _interp(rot, [(0, 0), (0.7, -20), (1, 100)]),
    }


def _profile_reach(t: float) -> dict:
    r = (math.sin(2 * math.pi * t * 0.5) + 1) / 2  # 0→1→0 cycle
    return {
        "spine_bend": 5 + r * 20,
        "head_pitch": r * 15 - 5,
        "head_yaw": r * 10,
        "l_shoulder_x": -10 - r * 5,
        "r_shoulder_x": -10 - r * 90,
        "l_elbow": 15,
        "r_elbow": max(0, 20 - r * 20),
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 2, "r_knee": 2,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": r * 5,
        "hip_y_offset": 0,
    }


def _profile_turn(t: float) -> dict:
    cycle = t * 0.7
    turn  = math.sin(2 * math.pi * cycle)
    return {
        "spine_bend": 3,
        "head_pitch": 2,
        "head_yaw": turn * 60,
        "l_shoulder_x": turn * 20 - 10,
        "r_shoulder_x": -turn * 20 - 10,
        "l_elbow": 20, "r_elbow": 20,
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 2, "r_knee": 2,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": turn * 10,
        "hip_y_offset": 0,
    }


def _profile_sit(t: float) -> dict:
    blink = _sine(t, 0.2, 1.5)
    return {
        "spine_bend": 5 + blink * 0.5,
        "head_pitch": 5 + blink,
        "head_yaw": _sine(t, 0.15, 4),
        "l_shoulder_x": -15,
        "r_shoulder_x": -15,
        "l_elbow": 60, "r_elbow": 60,
        "l_hip_x": 80, "r_hip_x": 80,
        "l_knee": 100, "r_knee": 100,
        "l_ankle": -80, "r_ankle": -80,
        "hip_sway": _sine(t, 0.12, 2),
        "hip_y_offset": 50,
    }


def _profile_climb(t: float) -> dict:
    cycle = t * 1.2
    arm   = math.sin(2 * math.pi * cycle)
    leg   = -arm
    return {
        "spine_bend": 20,
        "head_pitch": -15,
        "head_yaw": arm * 5,
        "l_shoulder_x": -60 + arm * 40,
        "r_shoulder_x": -60 - arm * 40,
        "l_elbow": 50 + arm * 20, "r_elbow": 50 - arm * 20,
        "l_hip_x": 30 + leg * 35, "r_hip_x": 30 - leg * 35,
        "l_knee": max(0, 40 + leg * 40), "r_knee": max(0, 40 - leg * 40),
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": arm * 5,
        "hip_y_offset": abs(math.sin(2 * math.pi * cycle * 2)) * -5,
    }


def _profile_swim(t: float) -> dict:
    cycle = t * 1.0
    arm   = math.sin(2 * math.pi * cycle)
    leg   = math.sin(2 * math.pi * cycle * 2)
    return {
        "spine_bend": 8 + arm * 5,
        "head_pitch": -10,
        "head_yaw": 0,
        "l_shoulder_x": -40 + arm * 85, "r_shoulder_x": -40 - arm * 85,
        "l_elbow": max(0, 20 - arm * 30), "r_elbow": max(0, 20 + arm * 30),
        "l_hip_x": leg * 20, "r_hip_x": -leg * 20,
        "l_knee": max(0, leg * 25), "r_knee": max(0, -leg * 25),
        "l_ankle": leg * 15, "r_ankle": -leg * 15,
        "hip_sway": arm * 8,
        "hip_y_offset": _sine(t, 0.6, 5),
    }


def _profile_sneak(t: float) -> dict:
    cycle = t * 1.0
    stride = math.sin(2 * math.pi * cycle)
    return {
        "spine_bend": 25,
        "head_pitch": -5,
        "head_yaw": stride * 8,
        "l_shoulder_x": stride * 20 - 10, "r_shoulder_x": -stride * 20 - 10,
        "l_elbow": 30, "r_elbow": 30,
        "l_hip_x": stride * 30, "r_hip_x": -stride * 30,
        "l_knee": max(0, -stride * 60) + 30, "r_knee": max(0, stride * 60) + 30,
        "l_ankle": stride * 10, "r_ankle": -stride * 10,
        "hip_sway": stride * 5,
        "hip_y_offset": 30,
    }


def _profile_throw(t: float) -> dict:
    norm = (math.sin(2 * math.pi * t * 0.8) + 1) / 2
    return {
        "spine_bend": 10 + norm * 15,
        "head_pitch": 0,
        "head_yaw": -norm * 20,
        "l_shoulder_x": -20,
        "r_shoulder_x": -20 - norm * 90,
        "l_elbow": 30, "r_elbow": max(0, 90 - norm * 90),
        "l_hip_x": norm * 15, "r_hip_x": -norm * 15,
        "l_knee": norm * 20, "r_knee": norm * 10,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": norm * 10 - 5,
        "hip_y_offset": 0,
    }



_ACTION_DISPATCH = {
    "walk":    _profile_walk,
    "run":     _profile_run,
    "wave":    _profile_wave,
    "dance":   _profile_dance,
    "fight":   _profile_fight,
    "reach":   _profile_reach,
    "turn":    _profile_turn,
    "sit":     _profile_sit,
    "climb":   _profile_climb,
    "swim":    _profile_swim,
    "sneak":   _profile_sneak,
    "throw":   _profile_throw,
    "idle":    _profile_idle,
}


# ---------------------------------------------------------------------------
# Daily-Life Action Profiles
# ---------------------------------------------------------------------------

def _profile_eat(t: float) -> dict:
    """Bring hand to mouth repeatedly — fork/spoon eating motion."""
    # Right arm cycles: resting → raised to mouth → back
    cycle = (math.sin(2 * math.pi * t * 0.9) + 1) / 2   # 0→1→0 per cycle
    phase2 = (math.sin(2 * math.pi * t * 0.9 + math.pi) + 1) / 2  # inverse
    chew_lean = _sine(t, 1.8, 1.5)    # subtle head bob for chewing
    return {
        "spine_bend":   8 + chew_lean * 0.5,
        "head_pitch":   10 + chew_lean * 3,   # look slightly down at food
        "head_yaw":     -5,                    # slight right gaze
        "l_shoulder_x": -20,
        "r_shoulder_x": -10 - cycle * 60,     # raise right arm to mouth
        "l_elbow": 40,
        "r_elbow": 80 - cycle * 50,           # extend when raising, flex when down
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 2,  "r_knee": 2,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": _sine(t, 0.15, 1.5),
        "hip_y_offset": 0,
    }


def _profile_sleep(t: float) -> dict:
    """Lying-down sleeping pose with slow breathing animation."""
    breathe = _sine(t, 0.2, 2.0)   # very slow breathing
    # Full horizontal pose: large hip_y_offset pushes body up (lying down)
    return {
        "spine_bend":   80 + breathe * 0.5,   # almost flat
        "head_pitch":   80,                    # head sideways
        "head_yaw":     20,
        "l_shoulder_x": 10 + breathe * 0.5,
        "r_shoulder_x": 10 + breathe * 0.5,
        "l_elbow": 20, "r_elbow": 20,
        "l_hip_x": 5,  "r_hip_x": 5,
        "l_knee": 15,  "r_knee": 15,
        "l_ankle": 0,  "r_ankle": 0,
        "hip_sway": 0,
        "hip_y_offset": 80 + breathe * 2,     # body shifted high = lying on ground
    }


def _profile_drink(t: float) -> dict:
    """Raise a cup/glass to drink — cyclical: rest → raise → sip → lower."""
    cycle = (math.sin(2 * math.pi * t * 0.55) + 1) / 2
    head_tilt = _interp(cycle, [(0, 0), (0.5, -15), (1, 0)])  # tilt back when sipping
    return {
        "spine_bend":   5 + cycle * 3,
        "head_pitch":   head_tilt,
        "head_yaw":     0,
        "l_shoulder_x": -15,
        "r_shoulder_x": -15 - cycle * 70,     # raise right arm
        "l_elbow": 25,
        "r_elbow": max(0, 80 - cycle * 60),   # arm extends forward to mouth
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 2,  "r_knee": 2,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": _sine(t, 0.1, 1.0),
        "hip_y_offset": 0,
    }


def _profile_read(t: float) -> dict:
    """Reading — both arms extended forward/down holding a book, head bowed."""
    page_turn = _sine(t, 0.08, 5)   # very slow occasional head pan (page turn)
    bob = _sine(t, 0.25, 0.8)       # tiny reading bob
    return {
        "spine_bend":   18 + bob * 0.5,
        "head_pitch":   20 + bob,             # look down at book
        "head_yaw":     page_turn,
        "l_shoulder_x": 25,                   # both arms forward holding book
        "r_shoulder_x": 25,
        "l_elbow": 55, "r_elbow": 55,
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 2,  "r_knee": 2,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": _sine(t, 0.12, 1.2),
        "hip_y_offset": 0,
    }


def _profile_phone(t: float) -> dict:
    """Talking on phone — right hand held to right ear, head slightly tilted."""
    nod = _sine(t, 0.4, 4)   # nodding while talking
    return {
        "spine_bend":   5,
        "head_pitch":   5 + nod * 0.5,
        "head_yaw":     15,                    # head turned slightly to right (phone side)
        "l_shoulder_x": -10,                   # left arm relaxed
        "r_shoulder_x": -30,                   # right arm raised to ear
        "l_elbow": 15,
        "r_elbow": 100,                        # high elbow flex — hand at ear
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 2,  "r_knee": 2,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": _sine(t, 0.2, 1.8),
        "hip_y_offset": 0,
    }


def _profile_bow(t: float) -> dict:
    """Respectful bow — torso bends forward then returns upright, looping."""
    # 0→1→0 cycle: 0=upright, 1=fully bowed
    cycle = (math.sin(2 * math.pi * t * 0.5 - math.pi / 2) + 1) / 2
    bend = cycle * 55
    return {
        "spine_bend":   bend,
        "head_pitch":   bend * 0.6,
        "head_yaw":     0,
        "l_shoulder_x": -bend * 0.4,
        "r_shoulder_x": -bend * 0.4,
        "l_elbow": 15 + bend * 0.3, "r_elbow": 15 + bend * 0.3,
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 5 + bend * 0.1, "r_knee": 5 + bend * 0.1,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": 0,
        "hip_y_offset": 0,
    }


def _profile_think(t: float) -> dict:
    """Thinking — right hand to chin, head tilted, occasional head scratch."""
    tilt = _sine(t, 0.15, 5)      # slow head sway while pondering
    eye_shift = _sine(t, 0.07, 8) # eyes shifting (head_yaw)
    return {
        "spine_bend":   8,
        "head_pitch":   8,
        "head_yaw":     tilt + eye_shift,
        "l_shoulder_x": -15,
        "r_shoulder_x": -10,                   # right arm bent up toward chin
        "l_elbow": 15,
        "r_elbow": 95,                          # high flex — hand near face
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 2,  "r_knee": 2,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": tilt * 0.5,
        "hip_y_offset": 0,
    }


def _profile_stretch(t: float) -> dict:
    """Full-body overhead stretch — arms rise overhead, spine extends back."""
    cycle = (math.sin(2 * math.pi * t * 0.4 - math.pi / 2) + 1) / 2
    arm_rise = cycle * 110
    spine_ext = cycle * 12
    leg_spread = cycle * 8
    return {
        "spine_bend":   -spine_ext,             # lean back during stretch
        "head_pitch":   -spine_ext * 0.6,
        "head_yaw":     0,
        "l_shoulder_x": -arm_rise,             # raise both arms overhead
        "r_shoulder_x": -arm_rise,
        "l_elbow": max(0, 10 - cycle * 10), "r_elbow": max(0, 10 - cycle * 10),
        "l_hip_x": -leg_spread, "r_hip_x": -leg_spread,
        "l_knee": 0, "r_knee": 0,
        "l_ankle": cycle * 8, "r_ankle": cycle * 8,  # rise slightly on toes
        "hip_sway": 0,
        "hip_y_offset": cycle * -8,             # slight upward shift as body stretches
    }


def _profile_push(t: float) -> dict:
    """Pushing motion — lean forward, both arms extend forward cyclically."""
    cycle = (math.sin(2 * math.pi * t * 0.7) + 1) / 2
    lean = 20 + cycle * 15
    arm_ext = cycle * 50
    return {
        "spine_bend":   lean,
        "head_pitch":   5,
        "head_yaw":     0,
        "l_shoulder_x": 30 + arm_ext,          # both arms forward
        "r_shoulder_x": 30 + arm_ext,
        "l_elbow": max(0, 40 - arm_ext * 0.8), "r_elbow": max(0, 40 - arm_ext * 0.8),
        "l_hip_x": -10, "r_hip_x": -10,
        "l_knee": 15 + cycle * 10, "r_knee": 15 + cycle * 10,
        "l_ankle": 10, "r_ankle": 10,
        "hip_sway": _sine(t, 0.7, 4),
        "hip_y_offset": cycle * 5,
    }


def _profile_pull(t: float) -> dict:
    """Pulling motion — lean back, arms pull toward body cyclically."""
    cycle = (math.sin(2 * math.pi * t * 0.7 + math.pi) + 1) / 2  # opposite phase to push
    lean = -5 - cycle * 10    # lean back
    arm_pull = cycle * 55
    return {
        "spine_bend":   lean,
        "head_pitch":   0,
        "head_yaw":     0,
        "l_shoulder_x": 55 - arm_pull,         # arms start extended then pull in
        "r_shoulder_x": 55 - arm_pull,
        "l_elbow": 10 + arm_pull * 0.8, "r_elbow": 10 + arm_pull * 0.8,
        "l_hip_x": 5, "r_hip_x": 5,
        "l_knee": 20 + cycle * 15, "r_knee": 20 + cycle * 15,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": _sine(t, 0.7, 3),
        "hip_y_offset": 5,
    }


# Register new actions in dispatch table
_ACTION_DISPATCH.update({
    "eat":     _profile_eat,
    "sleep":   _profile_sleep,
    "drink":   _profile_drink,
    "read":    _profile_read,
    "phone":   _profile_phone,
    "bow":     _profile_bow,
    "think":   _profile_think,
    "stretch": _profile_stretch,
    "push":    _profile_push,
    "pull":    _profile_pull,
})


def _get_pose(action: str, t: float, duration: float,
              actions: list = None) -> dict:
    """Return a pose dict for the given action(s) and time t.

    If ``actions`` is provided (a list of (action, weight) tuples from
    _classify_actions), poses are blended proportionally. Otherwise falls
    back to the single ``action`` string.
    """
    _one_shot = {"jump", "fall"}

    def _single_pose(act: str) -> dict:
        if act in _one_shot:
            fn = _profile_jump if act == "jump" else _profile_fall
            return fn(t, duration)
        return _ACTION_DISPATCH.get(act, _profile_idle)(t)

    if actions and len(actions) >= 2:
        # Multi-action blend
        pose = _single_pose(actions[0][0])
        for act, w in actions[1:]:
            other = _single_pose(act)
            # weight_a is the cumulative weight of `pose` vs `other`
            pose = _blend_poses(pose, other, actions[0][1])
        return pose

    return _single_pose(action)


# ---------------------------------------------------------------------------
# Forward Kinematics — 2D screen projection
# ---------------------------------------------------------------------------

def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def _rot2d(angle_deg: float, vec):
    """Rotate a 2-vector by angle_deg degrees."""
    a = _deg2rad(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return (c * vec[0] - s * vec[1], s * vec[0] + c * vec[1])


def _compute_joints(
    pose: dict,
    canvas_w: int,
    canvas_h: int,
    ground_y: int,
) -> dict:
    """
    Compute 2D screen positions using anatomically-correct forward kinematics.

    Conventions
    -----------
    - Y increases downward (screen coords).
    - 'spine_bend' tilts the whole torso forward (positive = lean forward).
    - Hip rotation ('hip_roll') tilts the pelvis left/right for weight transfer.
    - Leg angles follow a knee-remains-forward convention:
        l/r_hip_x  = thigh flexion (+ = forward).
        l/r_knee   = knee flexion (+ = bend backward, i.e. always >= 0).
        l/r_ankle  = ankle plantarflexion (+ = point toe down).
    - Arm angles:
        l/r_shoulder_x = shoulder flexion (+ forward, - backward).
        l/r_elbow      = elbow flexion (0 = straight, positive = bent).
    """
    S = _SKEL_SCALE
    cx = canvas_w // 2

    hip_y_off  = int(pose.get("hip_y_offset", 0))
    hip_sway   = pose.get("hip_sway", 0)        # lateral pelvis shift (px)
    hip_roll   = pose.get("hip_roll",  0)        # pelvis tilt angle (deg)
    spine_bend = pose.get("spine_bend", 0)       # forward lean of spine (deg)

    # ── 1. Root: pelvis centre ───────────────────────────────────────────
    # Ground reference: feet should always touch ground_y.
    # We place the hip at a fixed fraction of total height above ground.
    hips_x = cx + int(hip_sway)
    hips_y = ground_y - int(0.485 * S) + hip_y_off

    pos: dict = {"hips": (hips_x, hips_y)}

    # ── 2. Spine chain ──────────────────────────────────────────────────
    # Each vertebral segment tilts by a share of total spine_bend.
    # Direction: spine leans forward (slight x-offset) + pitch upward.
    spine_segs = [
        ("hips",   "spine",  spine_bend * 0.25, _bone_px("spine")),
        ("spine",  "spine1", spine_bend * 0.40, _bone_px("spine1")),
        ("spine1", "spine2", spine_bend * 0.35, _bone_px("spine2")),
        ("spine2", "neck",   0,                 _bone_px("neck")),
        ("neck",   "head",   pose.get("head_pitch", 0) * 0.6,  _bone_px("head")),
    ]
    acc_angle = 0.0
    for parent, child, delta, length in spine_segs:
        px, py = pos[parent]
        acc_angle += delta
        # Upward movement with forward lean (acc_angle tilts from vertical)
        dx = math.sin(math.radians(acc_angle)) * length
        dy = -math.cos(math.radians(acc_angle)) * length
        pos[child] = (int(px + dx), int(py + dy))

    # ── 3. Head yaw (lateral rotation) ──────────────────────────────────
    head_yaw = pose.get("head_yaw", 0)
    if head_yaw and "neck" in pos:
        nx, ny = pos["neck"]
        # Displacement proportional to head radius
        head_r = _bone_px("head") // 2
        pos["head"] = (int(nx + math.sin(math.radians(head_yaw)) * head_r),
                       pos["head"][1])

    # ── 4. Shoulder girdle ───────────────────────────────────────────────
    sp2_x, sp2_y = pos["spine2"]
    sh_half = _bone_px("l_shoulder")   # half shoulder width
    # Shoulders tilt with spine + mild counter-hip-roll
    sh_tilt = spine_bend * 0.1 + hip_roll * -0.2

    for side, sign in (("l", -1), ("r", 1)):
        # Horizontal offset (+/- shoulder width) plus slight tilt
        sh_x = int(sp2_x + sign * sh_half + math.sin(math.radians(sh_tilt)) * 6 * sign)
        sh_y = int(sp2_y + abs(math.sin(math.radians(sh_tilt))) * 4)
        pos[f"{side}_shoulder"] = (sh_x, sh_y)

        # ── Upper arm (humerus) ──────────────────────────────────────────
        # sh_angle: + = forward raise, - = backward swing; measured from vertical
        sh_angle = pose.get(f"{side}_shoulder_x", 0)
        ua_len   = _bone_px(f"{side}_elbow")
        # Arm hangs down (dy positive) and swings forward/back
        ua_dx = math.sin(math.radians(sh_angle)) * ua_len * sign
        ua_dy = math.cos(math.radians(abs(sh_angle) - 5)) * ua_len  # slight splay
        el_x = int(sh_x + ua_dx)
        el_y = int(sh_y + ua_dy)
        pos[f"{side}_elbow"] = (el_x, el_y)

        # ── Forearm (radius + ulna) ───────────────────────────────────────
        # el_angle: elbow flexion, 0 = straight, 90 = right-angle
        el_flex = max(0.0, pose.get(f"{side}_elbow", 10))  # always >= 0
        # Total forearm direction = shoulder direction + elbow flex
        fa_angle = sh_angle + el_flex * sign * 0.5
        fa_len   = _bone_px(f"{side}_wrist")
        fa_dx = math.sin(math.radians(fa_angle)) * fa_len * sign
        fa_dy = math.cos(math.radians(max(0, abs(fa_angle) - 5))) * fa_len
        pos[f"{side}_wrist"] = (int(el_x + fa_dx), int(el_y + fa_dy))

    # ── 5. Pelvis & legs ────────────────────────────────────────────────
    hip_half = _bone_px("l_hip")    # half pelvis width

    for side, sign in (("l", -1), ("r", 1)):
        # Hip joint position (on pelvis, with hip roll)
        roll_dy = math.sin(math.radians(hip_roll)) * hip_half
        hip_x   = int(hips_x + sign * hip_half)
        hip_y   = int(hips_y + sign * roll_dy)
        pos[f"{side}_hip"] = (hip_x, hip_y)

        # ── Thigh (femur) ────────────────────────────────────────────────
        # hip_angle: + = forward swing (hip flexion), - = backward extension
        hip_angle = pose.get(f"{side}_hip_x", 0)
        th_len    = _bone_px(f"{side}_knee")
        # Add spine lean to leg so the whole body leans together
        total_hip = hip_angle + spine_bend * 0.15
        th_dx = math.sin(math.radians(total_hip)) * th_len
        th_dy = math.cos(math.radians(total_hip)) * th_len   # grows downward
        kn_x  = int(hip_x + th_dx)
        kn_y  = int(hip_y + th_dy)
        pos[f"{side}_knee"] = (kn_x, kn_y)

        # ── Shin (tibia + fibula) ────────────────────────────────────────
        kn_flex = max(0.0, pose.get(f"{side}_knee", 0))  # always >= 0
        sh_len  = _bone_px(f"{side}_ankle")
        # Shin direction = thigh direction + knee flex (bends backward)
        shin_angle = total_hip - kn_flex
        an_dx  = math.sin(math.radians(shin_angle)) * sh_len
        an_dy  = math.cos(math.radians(shin_angle)) * sh_len
        an_x   = int(kn_x + an_dx)
        an_y   = int(kn_y + an_dy)
        pos[f"{side}_ankle"] = (an_x, an_y)

        # ── Foot ─────────────────────────────────────────────────────────
        an_flex = pose.get(f"{side}_ankle", 0)  # dorsiflexion(+) / planta(-)
        ft_len  = _bone_px(f"{side}_toe")
        # Foot extends forward from ankle; plantar-flex tilts foot downward
        foot_angle = shin_angle + an_flex - 90   # 90 offset so 0 = horizontal
        to_dx = math.cos(math.radians(foot_angle)) * ft_len * sign
        to_dy = math.sin(math.radians(foot_angle)) * ft_len
        pos[f"{side}_toe"] = (int(an_x + to_dx), int(an_y + to_dy))

    return pos


# ---------------------------------------------------------------------------
# Renderer — Realistic Bone Edition
# ---------------------------------------------------------------------------

# ── Realistic bone color palette ────────────────────────────────────────────
# All parts share the same ivory / warm-off-white bone family.
# Three tones simulate ambient + diffuse + highlight of real bone under light.
_C_BONE_BRIGHT  = (240, 228, 196)   # bright highlight on bone surface
_C_BONE_MID     = (200, 185, 148)   # mid-tone bone body
_C_BONE_SHADOW  = (120, 105,  75)   # cortex shadow / depth
_C_BONE_GLOW    = ( 50,  42,  22)   # dark warm shadow (sub-surface glow)
# Joints are rendered slightly whiter/bigger to simulate cartilage & epiphysis
_C_JOINT_HL     = (252, 244, 220)   # joint highlight
_C_JOINT_CART   = (195, 180, 140)   # cartilage body
_C_JOINT_SHADOW = ( 90,  78,  50)   # joint socket shadow

# Joint name → (fill_color, glow_color, circle_radius)
_JOINT_STYLES: dict = {
    # Skull — bigger, rounder, bright with visible socket shading
    "head":       (_C_BONE_BRIGHT, _C_BONE_GLOW,   16),
    "neck":       (_C_JOINT_CART,  _C_BONE_GLOW,    6),
    # Spine / pelvis
    "spine2":     (_C_JOINT_CART,  _C_BONE_GLOW,    7),
    "spine1":     (_C_JOINT_CART,  _C_BONE_GLOW,    6),
    "spine":      (_C_JOINT_CART,  _C_BONE_GLOW,    6),
    "hips":       (_C_BONE_MID,    _C_BONE_GLOW,    9),
    # Arm joints
    "l_shoulder": (_C_JOINT_HL,    _C_BONE_GLOW,    8),
    "l_elbow":    (_C_JOINT_CART,  _C_BONE_GLOW,    7),
    "l_wrist":    (_C_JOINT_CART,  _C_BONE_GLOW,    5),
    "r_shoulder": (_C_JOINT_HL,    _C_BONE_GLOW,    8),
    "r_elbow":    (_C_JOINT_CART,  _C_BONE_GLOW,    7),
    "r_wrist":    (_C_JOINT_CART,  _C_BONE_GLOW,    5),
    # Leg joints
    "l_hip":      (_C_BONE_MID,    _C_BONE_GLOW,    7),
    "l_knee":     (_C_JOINT_HL,    _C_BONE_GLOW,    8),
    "l_ankle":    (_C_JOINT_CART,  _C_BONE_GLOW,    6),
    "l_toe":      (_C_JOINT_CART,  _C_BONE_GLOW,    4),
    "r_hip":      (_C_BONE_MID,    _C_BONE_GLOW,    7),
    "r_knee":     (_C_JOINT_HL,    _C_BONE_GLOW,    8),
    "r_ankle":    (_C_JOINT_CART,  _C_BONE_GLOW,    6),
    "r_toe":      (_C_JOINT_CART,  _C_BONE_GLOW,    4),
}

# Bone → (bone_color, shadow_color, main_half_width, shadow_width)
# main_half_width is used for the tapered polygon; bigger = thicker bone
_BONE_STYLES: dict = {
    # Spine — widest, most massive
    "spine":      (_C_BONE_MID,    _C_BONE_SHADOW,  7, 14),
    "spine1":     (_C_BONE_MID,    _C_BONE_SHADOW,  7, 14),
    "spine2":     (_C_BONE_MID,    _C_BONE_SHADOW,  6, 12),
    "neck":       (_C_BONE_BRIGHT, _C_BONE_SHADOW,  4,  9),
    "head":       (_C_BONE_BRIGHT, _C_BONE_SHADOW,  4,  9),
    # Arms — humerus thicker than radius
    "l_shoulder": (_C_BONE_MID,    _C_BONE_SHADOW,  6, 12),
    "l_elbow":    (_C_BONE_BRIGHT, _C_BONE_SHADOW,  5, 10),
    "l_wrist":    (_C_BONE_MID,    _C_BONE_SHADOW,  3,  7),
    "r_shoulder": (_C_BONE_MID,    _C_BONE_SHADOW,  6, 12),
    "r_elbow":    (_C_BONE_BRIGHT, _C_BONE_SHADOW,  5, 10),
    "r_wrist":    (_C_BONE_MID,    _C_BONE_SHADOW,  3,  7),
    # Legs — femur thickest, tibia medium, foot thinner
    "l_hip":      (_C_BONE_MID,    _C_BONE_SHADOW,  8, 16),
    "l_knee":     (_C_BONE_BRIGHT, _C_BONE_SHADOW,  7, 13),
    "l_ankle":    (_C_BONE_MID,    _C_BONE_SHADOW,  4,  9),
    "l_toe":      (_C_BONE_MID,    _C_BONE_SHADOW,  3,  6),
    "r_hip":      (_C_BONE_MID,    _C_BONE_SHADOW,  8, 16),
    "r_knee":     (_C_BONE_BRIGHT, _C_BONE_SHADOW,  7, 13),
    "r_ankle":    (_C_BONE_MID,    _C_BONE_SHADOW,  4,  9),
    "r_toe":      (_C_BONE_MID,    _C_BONE_SHADOW,  3,  6),
}


def _draw_bg(draw, w: int, h: int, ground_y: int) -> None:
    """Dark atmospheric background suitable for bone/anatomy viewing."""
    # Deep charcoal-black gradient — like a forensics/anatomy lab
    for y in range(h):
        frac = y / h
        rv = int(10 + frac * 8)
        gv = int(10 + frac * 8)
        bv = int(14 + frac * 10)
        draw.line([(0, y), (w, y)], fill=(rv, gv, bv))
    # Subtle floor
    draw.rectangle([(0, ground_y), (w, h)], fill=(12, 12, 16))
    # Very faint grid on floor only
    for x in range(0, w, 80):
        draw.line([(x, ground_y), (x, h)], fill=(22, 22, 28), width=1)
    # Ground line (subtle warm bone-coloured rim light)
    for off, a in [(-2, 15), (-1, 30), (0, 60), (1, 30), (2, 15)]:
        gv = 135 + a
        draw.line([(0, ground_y + off), (w, ground_y + off)],
                  fill=(gv // 2, gv // 2, gv // 3), width=1)


def _wrap_text(text: str, font, max_w: int, draw) -> list[str]:
    """Word-wrap text to list of lines fitting max_w pixels."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = (current + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_w:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]

# ---------------------------------------------------------------------------
# Visual Clarity Helpers
# ---------------------------------------------------------------------------

# Action accent colors (R, G, B) — used for badges, effects, etc.
_ACTION_ACCENTS: dict = {
    "run":    (255, 80,  40),   # fiery red-orange
    "walk":   (80,  200, 120),  # calm green
    "jump":   (120, 200, 255),  # sky
    "wave":   (255, 220,  60),  # golden yellow
    "dance":  (220,  80, 255),  # magenta
    "fight":  (255,  50,  50),  # bright red
    "fall":   (180, 140, 255),  # violet
    "reach":  (80,  220, 200),  # teal
    "turn":   (180, 180, 255),  # lavender
    "sit":    (100, 200, 255),  # light blue
    "climb":  (255, 160,  60),  # orange
    "swim":   ( 50, 170, 255),  # ocean blue
    "sneak":  (100, 255, 160),  # neon green
    "throw":  (255, 120,  60),  # amber
    "idle":   (160, 160, 200),  # muted
    "eat":    (255, 150,  50),
    "sleep":  ( 80,  80, 180),
    "drink":  ( 50, 200, 255),
    "read":   (220, 220, 180),
    "phone":  (100, 220, 100),
    "bow":    (180, 180, 180),
    "think":  (200, 150, 250),
    "stretch": (255, 200, 100),
    "push":   (255, 100,  50),
    "pull":   ( 50, 100, 255),
}

# Human-readable action labels with emoji
_ACTION_LABELS: dict = {
    "run":    "🏃 Running",
    "walk":   "🚶 Walking",
    "jump":   "🦘 Jumping",
    "wave":   "👋 Waving",
    "dance":  "💃 Dancing",
    "fight":  "🥊 Fighting",
    "fall":   "😱 Falling",
    "reach":  "🤲 Reaching",
    "turn":   "🔄 Turning",
    "sit":    "🪑 Sitting",
    "climb":  "🧗 Climbing",
    "swim":   "🏊 Swimming",
    "sneak":  "🕵️ Sneaking",
    "throw":  "🎯 Throwing",
    "idle":   "🧍 Standing",
    "eat":    "🍔 Eating",
    "sleep":  "💤 Sleeping",
    "drink":  "🥤 Drinking",
    "read":   "📖 Reading",
    "phone":  "📱 Calling",
    "bow":    "🙇 Bowing",
    "think":  "🤔 Thinking",
    "stretch": "🧘 Stretching",
    "push":   "🤚 Pushing",
    "pull":   "✊ Pulling",
}


def _draw_speech_bubble(
    draw,
    head_pos: tuple,
    canvas_w: int,
    canvas_h: int,
    text: str,
    t: float,
    font,
) -> None:
    """
    Draw an animated speech bubble above the head, with:
    - Smooth pop-in animation during the first 0.5s
    - Rounded rectangle background with border
    - Multi-line wrapped text
    - Triangle tail pointing toward the head
    - Subtle pulsing border alpha
    """
    from PIL import ImageDraw
    hx, hy = head_pos

    # Pop-in scale (0→1 over first 0.4 s)
    scale  = min(1.0, t / 0.4) if t < 0.4 else 1.0
    # Pulse: very subtle opacity flicker on border
    pulse  = int(210 + 30 * math.sin(2 * math.pi * t * 0.8))

    # Bubble dimensions
    max_bubble_w = 280
    pad_x, pad_y = 18, 12
    line_h = 22
    lines  = _wrap_text(text, font, max_bubble_w - pad_x * 2, draw)
    text_w = max(
        draw.textbbox((0, 0), ln, font=font)[2]
        for ln in lines
    )
    text_h = len(lines) * line_h

    bw = text_w + pad_x * 2
    bh = text_h + pad_y * 2

    # Apply pop-in scale from head center
    bw = int(bw * scale)
    bh = int(bh * scale)
    if bw < 4 or bh < 4:
        return

    # Position — prefer right-of-center, above head
    bx = min(canvas_w - bw - 12, max(12, hx - bw // 2 + 40))
    by = max(10, hy - bh - 46)

    # Tail tip (connector to head)
    tail_x, tail_y = hx + 6, hy - 8

    # Shadow
    shadow_off = 4
    draw.rounded_rectangle(
        (bx + shadow_off, by + shadow_off, bx + bw + shadow_off, by + bh + shadow_off),
        radius=14,
        fill=(0, 0, 0, 80) if False else (5, 5, 15),  # PIL RGBA not supported in RGB mode
    )

    # Bubble fill (soft translucent navy)
    draw.rounded_rectangle(
        (bx, by, bx + bw, by + bh),
        radius=14,
        fill=(18, 22, 48),
    )

    # Bubble border (pulsing)
    border_col = (80, 130, pulse)
    draw.rounded_rectangle(
        (bx, by, bx + bw, by + bh),
        radius=14,
        outline=border_col,
        width=2,
    )

    # Tail (triangle pointing from bubble bottom to head)
    if scale >= 0.8:  # Only draw tail when bubble is mostly visible
        bx_mid = bx + bw // 2
        tail_pts = [
            (bx_mid - 8, by + bh),
            (bx_mid + 8, by + bh),
            (tail_x, tail_y),
        ]
        draw.polygon(tail_pts, fill=(18, 22, 48))
        draw.line([(bx_mid - 8, by + bh), (tail_x, tail_y)], fill=border_col, width=2)
        draw.line([(bx_mid + 8, by + bh), (tail_x, tail_y)], fill=border_col, width=2)

    # Text lines
    ty = by + pad_y
    for line in lines:
        lbbox = draw.textbbox((0, 0), line, font=font)
        lw = lbbox[2] - lbbox[0]
        # Center text in bubble
        draw.text((bx + (bw - lw) // 2, ty), line, fill=(220, 230, 255), font=font)
        ty += line_h


def _render_frame(
    pose: dict,
    canvas_w: int,
    canvas_h: int,
    ground_y: int,
    prompt: str = "",
    t: float = 0.0,
) -> np.ndarray:
    """Render a single animation frame with premium visuals and speech bubble."""
    from PIL import Image, ImageDraw, ImageFont
    img  = Image.new("RGB", (canvas_w, canvas_h), (6, 6, 20))
    draw = ImageDraw.Draw(img)

    # ── Background ─────────────────────────────────────────────────────────
    _draw_bg(draw, canvas_w, canvas_h, ground_y)

    # ── Compute joint positions ────────────────────────────────────────────
    jpos = _compute_joints(pose, canvas_w, canvas_h, ground_y)
    hx, hy = jpos.get("hips", (canvas_w // 2, ground_y))

    # ── Ground shadow ellipse ──────────────────────────────────────────────
    shadow_rx = 40
    for i, (rx_off, col) in enumerate([
        (12, (15, 15, 38)), (6, (20, 18, 45)), (0, (28, 24, 60))
    ]):
        draw.ellipse(
            (hx - shadow_rx + rx_off, ground_y,
             hx + shadow_rx - rx_off, ground_y + 10),
            fill=col,
        )

    # ── Bones (glow layer first, then sharp) ──────────────────────────────
    for parent, child in SKELETON_BONES:
        if parent not in jpos or child not in jpos:
            continue
        px, py = jpos[parent]
        cx, cy = jpos[child]
        style  = _BONE_STYLES.get(child, (_C_SPINE, _C_GLOW_S, 4, 10))
        bone_col, glow_col, main_w, glow_w = style

        # Outer glow
        draw.line([(px, py), (cx, cy)], fill=glow_col, width=glow_w)
        # Mid soft
        mid = tuple(min(255, int(c * 0.55)) for c in bone_col)
        draw.line([(px, py), (cx, cy)], fill=mid, width=main_w + 2)
        # Core sharp line
        draw.line([(px, py), (cx, cy)], fill=bone_col, width=main_w)

    # ── Joints (3-layer: glow ring, outer circle, inner highlight) ─────────
    for jname, (jx, jy) in jpos.items():
        style = _JOINT_STYLES.get(jname, (_C_SPINE, _C_GLOW_S, 4))
        fill_col, glow_col, r = style

        # Head gets a circle face with facial features
        if jname == "head":
            # Outer glow ring
            draw.ellipse((jx - r - 4, jy - r - 4, jx + r + 4, jy + r + 4), fill=glow_col)
            # Skull circle
            draw.ellipse((jx - r, jy - r, jx + r, jy + r), fill=(30, 30, 50))
            # Head outline
            draw.ellipse((jx - r, jy - r, jx + r, jy + r), outline=fill_col, width=2)
            # Eyes
            eye_y = jy - 2
            for ex in (jx - 4, jx + 4):
                draw.ellipse((ex - 2, eye_y - 2, ex + 2, eye_y + 2), fill=fill_col)
            # Smile dot
            draw.ellipse((jx - 1, jy + 3, jx + 1, jy + 5), fill=fill_col)
        else:
            # Glow
            draw.ellipse((jx - r - 3, jy - r - 3, jx + r + 3, jy + r + 3), fill=glow_col)
            # Outer shell
            draw.ellipse((jx - r, jy - r, jx + r, jy + r), fill=fill_col)
            # Inner highlight (brighter center)
            inner = tuple(min(255, c + 80) for c in fill_col)
            ir = max(1, r - 2)
            draw.ellipse((jx - ir, jy - ir - 1, jx + ir, jy + ir - 1), fill=inner)

    # ── Action label (bottom-left) ─────────────────────────────────────────
    try:
        font_label = ImageFont.truetype("arial.ttf", 15)
        font_bubble = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font_label  = ImageFont.load_default()
        font_bubble = font_label

    # ── Speech bubble with prompt ──────────────────────────────────────────
    head_pos = jpos.get("head", (canvas_w // 2, 80))
    if prompt:
        _draw_speech_bubble(draw, head_pos, canvas_w, canvas_h, prompt, t, font_bubble)

    return np.array(img)


# ---------------------------------------------------------------------------
# Visual Clarity Helpers
# ---------------------------------------------------------------------------

# Action accent colors (R, G, B) — used for badges, effects, etc.
_ACTION_ACCENTS: dict = {
    "run":    (255, 80,  40),   # fiery red-orange
    "walk":   (80,  200, 120),  # calm green
    "jump":   (120, 200, 255),  # sky
    "wave":   (255, 220,  60),  # golden yellow
    "dance":  (220,  80, 255),  # magenta
    "fight":  (255,  50,  50),  # bright red
    "fall":   (180, 140, 255),  # violet
    "reach":  (80,  220, 200),  # teal
    "turn":   (180, 180, 255),  # lavender
    "sit":    (100, 200, 255),  # light blue
    "climb":  (255, 160,  60),  # orange
    "swim":   ( 50, 170, 255),  # ocean blue
    "sneak":  (100, 255, 160),  # neon green
    "throw":  (255, 120,  60),  # amber
    "idle":   (160, 160, 200),  # muted
}

# Human-readable action labels with emoji
_ACTION_LABELS: dict = {
    "run":    "🏃 Running",
    "walk":   "🚶 Walking",
    "jump":   "🦘 Jumping",
    "wave":   "👋 Waving",
    "dance":  "💃 Dancing",
    "fight":  "🥊 Fighting",
    "fall":   "😱 Falling",
    "reach":  "🤲 Reaching",
    "turn":   "🔄 Turning",
    "sit":    "🪑 Sitting",
    "climb":  "🧗 Climbing",
    "swim":   "🏊 Swimming",
    "sneak":  "🕵️ Sneaking",
    "throw":  "🎯 Throwing",
    "idle":   "🧍 Standing",
}


def _draw_action_badge(
    draw, action: str, canvas_w: int, canvas_h: int, t: float, font
) -> None:
    """Draw a prominent pill badge in the top-left corner showing the action name."""
    accent = _ACTION_ACCENTS.get(action, (200, 200, 200))
    label  = _ACTION_LABELS.get(action, action.capitalize())

    # Bounce-in on first 0.35 s
    scale = min(1.0, t / 0.35) if t < 0.35 else 1.0
    # Pulse outer ring
    pulse = int(80 + 40 * math.sin(2 * math.pi * t * 0.6))

    bbox  = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 14, 8
    bw = int((tw + pad_x * 2) * scale)
    bh = int((th + pad_y * 2) * scale)
    bx, by = 14, 14

    if bw < 8 or bh < 8:
        return

    rad = min(14, bw // 2, bh // 2)
    # Dark outer glow
    glow = tuple(c // 5 for c in accent)
    draw.rounded_rectangle((bx - 3, by - 3, bx + bw + 3, by + bh + 3),
                           radius=min(16, bw // 2, bh // 2), fill=glow)
    # Solid dark fill
    draw.rounded_rectangle((bx, by, bx + bw, by + bh),
                           radius=rad, fill=(12, 14, 34))
    # Accent border that pulses slightly
    border = tuple(min(255, c + pulse // 4) for c in accent)
    draw.rounded_rectangle((bx, by, bx + bw, by + bh),
                           radius=rad, outline=border, width=2)
    # Left accent bar (only when there's enough height)
    if bh > 16:
        bar_y0 = by + 6
        bar_y1 = by + bh - 6
        if bar_y1 > bar_y0:
            draw.rounded_rectangle((bx, bar_y0, bx + 4, bar_y1),
                                   radius=2, fill=accent)
    # Text
    text_x = bx + 10 + 2  # after the accent bar
    text_y = by + max(0, (bh - th) // 2)
    draw.text((text_x, text_y), label, fill=accent, font=font)


def _draw_speed_lines(
    draw, action: str, jpos: dict, t: float, canvas_w: int, canvas_h: int
) -> None:
    """Draw horizontal speed lines for fast actions (run, fight, throw)."""
    if action not in ("run", "fight", "throw"):
        return
    accent = _ACTION_ACCENTS.get(action, (255, 80, 40))
    cx = jpos.get("hips", (canvas_w // 2, 0))[0]
    cy_ref = jpos.get("spine1", (0, canvas_h // 2))[1]

    rng = lambda: np.random.default_rng(seed=int(t * 100) % 1000)
    rnd = rng()
    for i in range(8):
        y  = cy_ref + rnd.integers(-100, 100)
        x1 = max(0, cx - rnd.integers(80, 200))
        x2 = cx - rnd.integers(10, 40)
        alpha = rnd.integers(40, 110)
        col = tuple(int(c * alpha / 255) for c in accent)
        if x1 < x2:
            draw.line([(x1, y), (x2, y)], fill=col,
                      width=max(1, rnd.integers(1, 3)))


def _draw_direction_arrow(
    draw, action: str, jpos: dict, canvas_w: int
) -> None:
    """Draw a directional movement arrow for locomotion actions."""
    if action not in ("walk", "run", "sneak", "climb"):
        return
    accent = _ACTION_ACCENTS.get(action, (200, 200, 200))
    hx, hy = jpos.get("hips", (canvas_w // 2, 300))
    # Arrow pointing right (canonical animation faces right)
    ax, ay = hx + 58, hy
    shaft_x = ax - 30
    draw.line([(shaft_x, ay), (ax, ay)], fill=accent, width=3)
    # Arrowhead
    draw.polygon([
        (ax, ay),
        (ax - 10, ay - 6),
        (ax - 10, ay + 6),
    ], fill=accent)


def _draw_footstep_dust(
    draw, action: str, jpos: dict, t: float, ground_y: int
) -> None:
    """Draw dust puff particles near the feet for energetic ground actions."""
    if action not in ("run", "jump", "fight", "walk"):
        return
    accent = _ACTION_ACCENTS.get(action, (200, 150, 100))
    for side in ("l", "r"):
        ax, ay = jpos.get(f"{side}_toe", jpos.get(f"{side}_ankle", (0, ground_y)))
        # Only draw dust near ground
        if ay < ground_y - 20:
            continue
        rnd = np.random.default_rng(seed=int(t * 100 + (0 if side == "l" else 50)) % 9999)
        for _ in range(4):
            dx = rnd.integers(-22, 22)
            dy = rnd.integers(-12, 4)
            r  = rnd.integers(3, 9)
            alpha = rnd.integers(30, 90)
            col = tuple(int(c * alpha / 255) for c in accent)
            draw.ellipse((ax + dx - r, ground_y + dy - r,
                          ax + dx + r, ground_y + dy + r), fill=col)


def _draw_energy_ring(
    draw, action: str, jpos: dict, t: float
) -> None:
    """Pulsing ring around the figure for high-energy actions."""
    if action not in ("dance", "fight", "jump"):
        return
    accent = _ACTION_ACCENTS.get(action, (200, 200, 200))
    hx, hy = jpos.get("hips", (0, 0))
    hy -= 60  # center on torso
    pulse_r = int(80 + 25 * math.sin(2 * math.pi * t * (2 if action == "fight" else 1.2)))
    alpha_ring = int(40 + 25 * abs(math.sin(2 * math.pi * t)))
    ring_col = tuple(int(c * alpha_ring / 255) for c in accent)
    draw.ellipse((hx - pulse_r, hy - pulse_r // 2,
                  hx + pulse_r, hy + pulse_r // 2),
                 outline=ring_col, width=2)


def _draw_swim_bubbles(draw, jpos: dict, t: float) -> None:
    """Rising bubbles for swim action."""
    hx, hy = jpos.get("head", (360, 200))
    rnd = np.random.default_rng(seed=int(t * 30) % 300)
    for _ in range(5):
        bx = hx + rnd.integers(-50, 50)
        by = hy - rnd.integers(5, 60)
        r  = rnd.integers(3, 8)
        draw.ellipse((bx - r, by - r, bx + r, by + r),
                     outline=(120, 200, 255), width=1)


def _draw_stars(draw, jpos: dict, t: float) -> None:
    """Sparkle stars around the figure for dance action."""
    hx, hy = jpos.get("hips", (360, 300))
    rnd = np.random.default_rng(seed=int(t * 20) % 200)
    for _ in range(6):
        sx = hx + rnd.integers(-100, 100)
        sy = hy + rnd.integers(-160, 20)
        r  = rnd.integers(2, 5)
        col = (
            rnd.integers(180, 255),
            rnd.integers(180, 255),
            rnd.integers(50, 255),
        )
        # 4-pointed star as two crosshair lines
        draw.line([(sx - r * 2, sy), (sx + r * 2, sy)], fill=col, width=1)
        draw.line([(sx, sy - r * 2), (sx, sy + r * 2)], fill=col, width=1)
        draw.ellipse((sx - 1, sy - 1, sx + 1, sy + 1), fill=col)


def _draw_action_glow(draw, action: str, jpos: dict, t: float) -> None:
    """Large diffuse glow halo around active joints for jump/fall."""
    if action not in ("jump", "fall"):
        return
    accent = _ACTION_ACCENTS.get(action, (200, 200, 255))
    for jname in ("l_wrist", "r_wrist", "l_ankle", "r_ankle"):
        if jname not in jpos:
            continue
        jx, jy = jpos[jname]
        pulse = int(18 + 8 * math.sin(2 * math.pi * t * 3))
        col = tuple(int(c * 0.25) for c in accent)
        draw.ellipse((jx - pulse, jy - pulse, jx + pulse, jy + pulse), fill=col)


def _render_frame(
    pose: dict,
    canvas_w: int,
    canvas_h: int,
    ground_y: int,
    prompt: str = "",
    t: float = 0.0,
    action: str = "idle",
) -> np.ndarray:
    """Render a single animation frame with premium visuals, effects and speech bubble."""
    from PIL import Image, ImageDraw, ImageFont
    img  = Image.new("RGB", (canvas_w, canvas_h), (6, 6, 20))
    draw = ImageDraw.Draw(img)

    # ── Background ─────────────────────────────────────────────────────────
    _draw_bg(draw, canvas_w, canvas_h, ground_y)

    # ── Compute joint positions ────────────────────────────────────────────
    jpos = _compute_joints(pose, canvas_w, canvas_h, ground_y)
    hx, hy = jpos.get("hips", (canvas_w // 2, ground_y))

    # ── Action-specific background effects (drawn before figure) ──────────
    _draw_speed_lines(draw, action, jpos, t, canvas_w, canvas_h)
    _draw_energy_ring(draw, action, jpos, t)
    _draw_action_glow(draw, action, jpos, t)

    # ── Ground shadow ellipse ──────────────────────────────────────────────
    shadow_rx = 40
    for rx_off, col in [
        (12, (15, 15, 38)), (6, (20, 18, 45)), (0, (28, 24, 60))
    ]:
        draw.ellipse(
            (hx - shadow_rx + rx_off, ground_y,
             hx + shadow_rx - rx_off, ground_y + 10),
            fill=col,
        )

    # ── Footstep dust (behind figure) ─────────────────────────────────────
    _draw_footstep_dust(draw, action, jpos, t, ground_y)

    # ── Bones (glow + tapered muscle-mass polygons with bone shading) ──────
    for parent, child in SKELETON_BONES:
        if parent not in jpos or child not in jpos:
            continue
        px, py = jpos[parent]
        cx, cy = jpos[child]
        style  = _BONE_STYLES.get(child, (_C_BONE_MID, _C_BONE_SHADOW, 5, 11))
        bone_col, shadow_col, main_w, glow_w = style

        # Outer shadow/ambient-occlusion line (stays as thick line)
        draw.line([(px, py), (cx, cy)], fill=shadow_col, width=glow_w)

        L = math.hypot(cx - px, cy - py)
        if L < 1e-3:
            continue
        nx, ny = -(cy - py) / L, (cx - px) / L

        # Mid cortex layer — tapers from parent to child (wider at joint end)
        ws_m, we_m = main_w * 1.7, main_w * 0.9
        mid_col = tuple(max(0, c - 30) for c in bone_col)
        poly_m = [
            (px + nx * ws_m, py + ny * ws_m),
            (cx + nx * we_m, cy + ny * we_m),
            (cx - nx * we_m, cy - ny * we_m),
            (px - nx * ws_m, py - ny * ws_m),
        ]
        draw.polygon(poly_m, fill=mid_col)

        # Core bright surface layer — narrower, simulates bone cortex surface
        ws_c, we_c = main_w * 1.0, main_w * 0.5
        poly_c = [
            (px + nx * ws_c, py + ny * ws_c),
            (cx + nx * we_c, cy + ny * we_c),
            (cx - nx * we_c, cy - ny * we_c),
            (px - nx * ws_c, py - ny * ws_c),
        ]
        draw.polygon(poly_c, fill=bone_col)

        # Bright specular ridge (top edge highlight — one pixel line)
        hl = tuple(min(255, c + 40) for c in bone_col)
        draw.line(
            [(int(px + nx * ws_c * 0.6), int(py + ny * ws_c * 0.6)),
             (int(cx + nx * we_c * 0.6), int(cy + ny * we_c * 0.6))],
            fill=hl, width=1,
        )

    # ── Joints — realistic bone epiphyses & skull ─────────────────────────
    for jname, (jx, jy) in jpos.items():
        style = _JOINT_STYLES.get(jname, (_C_BONE_MID, _C_BONE_GLOW, 4))
        fill_col, shadow_col, r = style

        if jname == "head":
            # ── Skull / Cranium ──────────────────────────────────────────
            # Outer dark halo (sub-surface / ambient occlusion)
            draw.ellipse((jx - r - 3, jy - r - 3, jx + r + 3, jy + r + 3),
                         fill=shadow_col)
            # Cranium body — off-white ivory
            draw.ellipse((jx - r, jy - r, jx + r, jy + r),
                         fill=_C_BONE_MID)
            # Upper highlight (bright dome simulating top-light)
            hr = max(4, r - 2)
            draw.ellipse((jx - hr, jy - r + 1, jx + hr, jy - 2),
                         fill=_C_BONE_BRIGHT)
            # Cranium outline
            draw.ellipse((jx - r, jy - r, jx + r, jy + r),
                         outline=_C_BONE_SHADOW, width=2)
            # Eye sockets (dark empty orbital cavities)
            eye_y = jy + 1
            for ex_off in (-r // 3, r // 3):
                ex = jx + ex_off
                es = max(2, r // 4)
                draw.ellipse((ex - es, eye_y - es, ex + es, eye_y + es),
                             fill=_C_BONE_GLOW)
            # Nasal aperture (small dark triangle below eyes)
            draw.polygon([
                (jx,     eye_y + r // 4),
                (jx - 2, eye_y + r // 2),
                (jx + 2, eye_y + r // 2),
            ], fill=_C_BONE_GLOW)
        else:
            # ── Ball-and-socket / hinge joints ──────────────────────────
            # Outer shadow ring (depth / socket effect)
            draw.ellipse((jx - r - 2, jy - r - 2, jx + r + 2, jy + r + 2),
                         fill=shadow_col)
            # Joint ball body
            draw.ellipse((jx - r, jy - r, jx + r, jy + r), fill=fill_col)
            # Specular highlight (top-left, simulating overhead light)
            ir = max(1, r - 3)
            hl_col = tuple(min(255, c + 55) for c in fill_col)
            draw.ellipse((jx - ir, jy - r + 1, jx + 1, jy - 1), fill=hl_col)
            # Bottom shadow arc (opposite side from light)
            sh_col = tuple(max(0, c - 50) for c in fill_col)
            draw.arc((jx - r + 1, jy + 1, jx + r - 1, jy + r - 1),
                     start=20, end=160, fill=sh_col, width=2)

    # ── Action-specific overlay effects (drawn OVER figure) ───────────────
    if action == "swim":
        _draw_swim_bubbles(draw, jpos, t)
    if action == "dance":
        _draw_stars(draw, jpos, t)
    _draw_direction_arrow(draw, action, jpos, canvas_w)

    # ── Fonts ──────────────────────────────────────────────────────────────
    try:
        font_badge  = ImageFont.truetype("arial.ttf", 17)
        font_bubble = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font_badge  = ImageFont.load_default()
        font_bubble = font_badge

    # ── Action badge (top-left) ────────────────────────────────────────────
    _draw_action_badge(draw, action, canvas_w, canvas_h, t, font_badge)

    # ── Speech bubble with prompt ──────────────────────────────────────────
    head_pos = jpos.get("head", (canvas_w // 2, 80))
    if prompt:
        _draw_speech_bubble(draw, head_pos, canvas_w, canvas_h, prompt, t, font_bubble)

    return np.array(img)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_animation(
    hint: str,
    output_dir: str = "data/generated",
    fps: int = 30,
    max_duration: float = 20.0,
) -> str:
    """Generate a realistic procedural skeleton animation for the given text prompt.

    Parameters
    ----------
    hint : str
        Natural-language description of the desired action.
    output_dir : str
        Directory to place the output MP4.
    fps : int
        Frames per second (default 30).
    max_duration : float
        Maximum clip duration in seconds (capped at 30 s).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    filename  = sanitize_hint(hint) + ".mp4"
    out_path  = os.path.join(output_dir, filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    # Parse actions list; pick primary action for loop logic / visuals
    actions  = _classify_actions(hint)
    primary  = actions[0][0]
    duration = min(float(max_duration), 30.0)

    # Looping actions get a shorter natural-cycle then we loop the clip
    loop_actions = {"walk", "run", "dance", "fight", "climb", "swim", "sneak", "idle", "wave", "drink", "eat", "read"}
    if primary in loop_actions and len(actions) == 1:
        # If single looping action, shorten cycle to save render time, then loop video
        duration = min(duration, 8.0) 

    canvas_w, canvas_h = 720, 540
    ground_y = canvas_h - 60

    times  = np.linspace(0, duration, int(duration * fps))
    frames = []

    print(f"[animation_generator] actions={actions} duration={duration}s frames={len(times)}")

    for t in times:
        pose  = _get_pose(primary, float(t), duration, actions=actions)
        frame = _render_frame(pose, canvas_w, canvas_h, ground_y, prompt=hint, t=float(t), action=primary)
        frames.append(frame)

    # Assemble video
    try:
        from moviepy import ImageSequenceClip
    except ImportError:
        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            print("[animation_generator] moviepy not available; creating empty file.")
            open(out_path, "wb").close()
            return out_path

    clip = ImageSequenceClip(frames, fps=fps)

    # Loop to fill full requested length
    full_duration = min(float(max_duration), 30.0)
    if clip.duration < full_duration and primary in loop_actions and len(actions) == 1:
        from moviepy import concatenate_videoclips  # type: ignore
        reps = math.ceil(full_duration / clip.duration)
        clip = concatenate_videoclips([clip] * reps).subclipped(0, full_duration)

    clip.write_videofile(out_path, codec="libx264", audio=False, logger=None)
    clip.close()
    print(f"[animation_generator] Written → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a realistic stick-figure animation.")
    parser.add_argument("hint", help="Text prompt describing the animation")
    parser.add_argument("--output-dir", "-o", default="data/generated")
    parser.add_argument("--duration", "-d", type=float, default=10.0)
    args = parser.parse_args()
    out  = create_animation(args.hint, args.output_dir, max_duration=args.duration)
    print(f"animation generated: {out}")
