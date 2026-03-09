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
    "run":    ["run", "sprint", "jog", "race", "chase", "rush", "flee", "dash"],
    "walk":   ["walk", "stroll", "march", "hike", "pace", "wander", "step"],
    "jump":   ["jump", "leap", "hop", "bounce", "skip", "vault", "spring"],
    "wave":   ["wave", "hello", "hi ", "hey", "greet", "goodbye", "bye"],
    "dance":  ["dance", "party", "celebrate", "groove", "spin", "twirl", "sway", "rhythm", "happy", "joy"],
    "fight":  ["fight", "punch", "kick", "attack", "battle", "combat", "strike", "hit", "defend"],
    "fall":   ["fall", "trip", "slip", "tumble", "collapse", "drop", "faint"],
    "reach":  ["reach", "grab", "pick", "lift", "pull", "push", "stretch", "extend", "take", "get"],
    "turn":   ["turn", "rotate", "spin", "look around", "look back", "pivot"],
    "sit":    ["sit", "seat", "rest", "relax", "couch", "chair", "bench", "kneel", "crouch"],
    "climb":  ["climb", "crawl", "scale", "ascend"],
    "swim":   ["swim", "float", "dive", "splash"],
    "sneak":  ["sneak", "creep", "tiptoe", "stealth"],
    "throw":  ["throw", "toss", "fling", "pitch", "hurl"],
    "idle":   [],  # fallback
}


def _classify_action(prompt: str) -> str:
    low = prompt.lower()
    for action, keywords in _ACTION_KEYWORDS.items():
        for kw in keywords:
            if kw in low:
                return action
    # Semantic fallback using STS model
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
        return labels[int(scores.argmax())]
    except Exception:
        return "idle"


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
_BONE_LEN = {
    ("hips", "spine"):      0.15,
    ("spine", "spine1"):    0.15,
    ("spine1", "spine2"):   0.15,
    ("spine2", "neck"):     0.12,
    ("neck", "head"):       0.14,
    ("spine2", "l_shoulder"): 0.18,
    ("l_shoulder", "l_elbow"): 0.22,
    ("l_elbow", "l_wrist"):  0.20,
    ("spine2", "r_shoulder"): 0.18,
    ("r_shoulder", "r_elbow"): 0.22,
    ("r_elbow", "r_wrist"):  0.20,
    ("hips", "l_hip"):     0.12,
    ("l_hip", "l_knee"):   0.28,
    ("l_knee", "l_ankle"): 0.26,
    ("l_ankle", "l_toe"):  0.10,
    ("hips", "r_hip"):     0.12,
    ("r_hip", "r_knee"):   0.28,
    ("r_knee", "r_ankle"): 0.26,
    ("r_ankle", "r_toe"):  0.10,
}

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
    breathe = _sine(t, 0.25, 1.5)
    head_nod = _sine(t, 0.20, 2.0)
    sway = _sine(t, 0.15, 0.8)
    return {
        "spine_bend": breathe * 0.3,
        "head_pitch": head_nod,
        "head_yaw": sway * 2.5,
        "l_shoulder_x": -5 + breathe * 0.5,
        "r_shoulder_x": -5 + breathe * 0.5,
        "l_elbow": 10,
        "r_elbow": 10,
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 2, "r_knee": 2,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": sway * 1.5,
        "hip_y_offset": 0,
    }


def _profile_walk(t: float) -> dict:
    cycle = t * 1.6  # ~1.6 strides/s
    stride = math.sin(2 * math.pi * cycle)
    arm    = -stride  # arms opposite to legs
    return {
        "spine_bend": 3 + _sine(t, 0.8 * 2, 0.8),
        "head_pitch": 2,
        "head_yaw": stride * 3,
        "l_shoulder_x": arm * 28,
        "r_shoulder_x": -arm * 28,
        "l_elbow": 25 + arm * 10,
        "r_elbow": 25 - arm * 10,
        "l_hip_x": stride * 30,
        "r_hip_x": -stride * 30,
        "l_knee": max(0, -stride * 40),
        "r_knee": max(0, stride * 40),
        "l_ankle": stride * 15,
        "r_ankle": -stride * 15,
        "hip_sway": stride * 4,
        "hip_y_offset": abs(math.sin(2 * math.pi * cycle * 2)) * -3,
    }


def _profile_run(t: float) -> dict:
    cycle = t * 2.8  # fast stride
    stride = math.sin(2 * math.pi * cycle)
    arm    = -stride
    return {
        "spine_bend": 12 + _sine(t, 2.8 * 2, 1.5),
        "head_pitch": 5,
        "head_yaw": stride * 4,
        "l_shoulder_x": arm * 55,
        "r_shoulder_x": -arm * 55,
        "l_elbow": 70 + arm * 15,
        "r_elbow": 70 - arm * 15,
        "l_hip_x": stride * 55,
        "r_hip_x": -stride * 55,
        "l_knee": max(0, -stride * 80),
        "r_knee": max(0, stride * 80),
        "l_ankle": stride * 25,
        "r_ankle": -stride * 25,
        "hip_sway": stride * 6,
        "hip_y_offset": abs(math.sin(2 * math.pi * cycle * 2)) * -8,
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
    "walk":  _profile_walk,
    "run":   _profile_run,
    "wave":  _profile_wave,
    "dance": _profile_dance,
    "fight": _profile_fight,
    "reach": _profile_reach,
    "turn":  _profile_turn,
    "sit":   _profile_sit,
    "climb": _profile_climb,
    "swim":  _profile_swim,
    "sneak": _profile_sneak,
    "throw": _profile_throw,
    "idle":  _profile_idle,
}


def _get_pose(action: str, t: float, duration: float) -> dict:
    """Return a pose dict for the given action and time t."""
    if action in ("jump", "fall"):
        fn = _profile_jump if action == "jump" else _profile_fall
        return fn(t, duration)
    elif action in _ACTION_DISPATCH:
        return _ACTION_DISPATCH[action](t)
    else:
        return _profile_idle(t)


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


def _compute_joints(pose: dict, canvas_w: int, canvas_h: int, ground_y: int) -> dict:
    """
    Compute 2D pixel positions of every joint for the given pose dict.
    Returns: {joint_name: (x, y)}
    """
    scale = _SKEL_SCALE
    cx = canvas_w // 2
    hip_offset_y = int(pose.get("hip_y_offset", 0))
    sway = int(pose.get("hip_sway", 0))

    # Root — hips
    hips_y = ground_y - int(0.40 * scale) + hip_offset_y
    hips_x = cx + sway

    pos = {"hips": (hips_x, hips_y)}

    # Spine chain (grows upward)
    spine_bend = pose.get("spine_bend", 0)
    chain_up = [
        ("hips", "spine",  spine_bend * 0.3,  _bone_px("spine")),
        ("spine",  "spine1", spine_bend * 0.4,  _bone_px("spine1")),
        ("spine1", "spine2", spine_bend * 0.3,  _bone_px("spine2")),
        ("spine2", "neck",   0,                  _bone_px("neck")),
        ("neck",   "head",   pose.get("head_pitch", 0), _bone_px("head")),
    ]
    for parent, child, angle, length in chain_up:
        px, py = pos[parent]
        # "up" direction is negative y, with a lateral tilt from angle
        dx, dy = _rot2d(angle, (0, -length))
        pos[child] = (int(px + dx), int(py + dy))

    # Head yaw — move head left/right relative to neck
    head_yaw = pose.get("head_yaw", 0)
    if head_yaw and "neck" in pos and "head" in pos:
        neck_x, neck_y = pos["neck"]
        _, hy = pos["head"]
        pos["head"] = (int(neck_x + head_yaw * 0.4), hy)

    # Arms
    sp2_x, sp2_y = pos["spine2"]
    shoulder_width = _bone_px("l_shoulder")

    for side in ("l", "r"):
        sign = -1 if side == "l" else 1
        sh_x = int(sp2_x + sign * shoulder_width)
        sh_y = sp2_y
        pos[f"{side}_shoulder"] = (sh_x, sh_y)

        elbow_len = _bone_px(f"{side}_elbow")
        sh_angle  = pose.get(f"{side}_shoulder_x", -10)
        edx, edy  = _rot2d(sh_angle, (0, elbow_len))
        el_x = int(sh_x + sign * abs(edx) + edx * (1 - abs(sign)) * 0.3)
        el_y = int(sh_y + edy)
        pos[f"{side}_elbow"] = (el_x, el_y)

        wrist_len = _bone_px(f"{side}_wrist")
        el_angle   = pose.get(f"{side}_elbow", 20)
        wdx, wdy   = _rot2d(el_angle * sign * 0.6 + sh_angle, (0, wrist_len))
        pos[f"{side}_wrist"] = (int(el_x + wdx), int(el_y + wdy))

    # Legs
    hip_width = _bone_px("l_hip")
    for side in ("l", "r"):
        sign = -1 if side == "l" else 1
        lh_x = int(hips_x + sign * hip_width)
        lh_y = hips_y
        pos[f"{side}_hip"] = (lh_x, lh_y)

        thigh_len = _bone_px(f"{side}_knee")
        hip_angle = pose.get(f"{side}_hip_x", 0)
        tdx, tdy  = _rot2d(hip_angle, (0, thigh_len))
        kn_x = int(lh_x + tdx)
        kn_y = int(lh_y + tdy)
        pos[f"{side}_knee"] = (kn_x, kn_y)

        shin_len  = _bone_px(f"{side}_ankle")
        kn_angle  = pose.get(f"{side}_knee", 0)
        sdx, sdy  = _rot2d(hip_angle + kn_angle * 0.5, (0, shin_len))
        an_x = int(kn_x + sdx)
        an_y = int(kn_y + sdy)
        pos[f"{side}_ankle"] = (an_x, an_y)

        toe_len  = _bone_px(f"{side}_toe")
        an_angle = pose.get(f"{side}_ankle", 0)
        # Toes extend horizontally forward from ankle
        pos[f"{side}_toe"] = (int(an_x + sign * toe_len + an_angle * 0.3), an_y + 4)

    return pos


# ---------------------------------------------------------------------------
# Renderer — Premium edition
# ---------------------------------------------------------------------------

# Color palette (R, G, B)
_C_SPINE   = (255, 210,  80)   # warm amber — spine/torso
_C_HEAD    = (255, 240, 160)   # bright cream — head/neck
_C_LEFT    = ( 90, 210, 255)   # sky blue — left limbs
_C_RIGHT   = (255, 110,  80)   # coral — right limbs
_C_GLOW_S  = ( 90,  70,  10)   # glow tint for spine
_C_GLOW_L  = (  0,  60,  90)   # glow tint for left
_C_GLOW_R  = ( 90,  30,   0)   # glow tint for right

# Joint name → (fill_color, glow_color, circle_radius)
_JOINT_STYLES: dict = {
    "head":       (_C_HEAD,  (60, 60, 20), 13),
    "neck":       (_C_HEAD,  (60, 60, 20),  5),
    "spine2":     (_C_SPINE, _C_GLOW_S,    5),
    "spine1":     (_C_SPINE, _C_GLOW_S,    4),
    "spine":      (_C_SPINE, _C_GLOW_S,    4),
    "hips":       (_C_SPINE, _C_GLOW_S,    6),
    "l_shoulder": (_C_LEFT,  _C_GLOW_L,    6),
    "l_elbow":    (_C_LEFT,  _C_GLOW_L,    5),
    "l_wrist":    (_C_LEFT,  _C_GLOW_L,    4),
    "r_shoulder": (_C_RIGHT, _C_GLOW_R,    6),
    "r_elbow":    (_C_RIGHT, _C_GLOW_R,    5),
    "r_wrist":    (_C_RIGHT, _C_GLOW_R,    4),
    "l_hip":      (_C_LEFT,  _C_GLOW_L,    5),
    "l_knee":     (_C_LEFT,  _C_GLOW_L,    5),
    "l_ankle":    (_C_LEFT,  _C_GLOW_L,    4),
    "l_toe":      (_C_LEFT,  _C_GLOW_L,    3),
    "r_hip":      (_C_RIGHT, _C_GLOW_R,    5),
    "r_knee":     (_C_RIGHT, _C_GLOW_R,    5),
    "r_ankle":    (_C_RIGHT, _C_GLOW_R,    4),
    "r_toe":      (_C_RIGHT, _C_GLOW_R,    3),
}

# Bone name (child joint) → (bone_color, glow_color, main_width, glow_width)
_BONE_STYLES: dict = {
    "spine":   (_C_SPINE, _C_GLOW_S, 5, 12),
    "spine1":  (_C_SPINE, _C_GLOW_S, 5, 12),
    "spine2":  (_C_SPINE, _C_GLOW_S, 5, 12),
    "neck":    (_C_HEAD,  (50, 50, 15), 4, 10),
    "head":    (_C_HEAD,  (50, 50, 15), 4, 10),
    "l_shoulder": (_C_LEFT,  _C_GLOW_L, 4, 10),
    "l_elbow":    (_C_LEFT,  _C_GLOW_L, 4, 10),
    "l_wrist":    (_C_LEFT,  _C_GLOW_L, 3,  8),
    "r_shoulder": (_C_RIGHT, _C_GLOW_R, 4, 10),
    "r_elbow":    (_C_RIGHT, _C_GLOW_R, 4, 10),
    "r_wrist":    (_C_RIGHT, _C_GLOW_R, 3,  8),
    "l_hip":      (_C_LEFT,  _C_GLOW_L, 5, 11),
    "l_knee":     (_C_LEFT,  _C_GLOW_L, 4, 10),
    "l_ankle":    (_C_LEFT,  _C_GLOW_L, 3,  8),
    "l_toe":      (_C_LEFT,  _C_GLOW_L, 2,  6),
    "r_hip":      (_C_RIGHT, _C_GLOW_R, 5, 11),
    "r_knee":     (_C_RIGHT, _C_GLOW_R, 4, 10),
    "r_ankle":    (_C_RIGHT, _C_GLOW_R, 3,  8),
    "r_toe":      (_C_RIGHT, _C_GLOW_R, 2,  6),
}


def _draw_bg(draw, w: int, h: int, ground_y: int) -> None:
    """3-band gradient + grid-line background."""
    # Top band — deep purple-navy
    for y in range(h):
        frac = y / h
        r = int(6  + frac * 14)
        g = int(4  + frac * 10)
        b = int(22 + frac * 30)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    # Floor highlight
    draw.rectangle([(0, ground_y - 1), (w, h)], fill=(10, 10, 28))
    # Faint grid lines
    for x in range(0, w, 80):
        draw.line([(x, ground_y), (x, h)], fill=(25, 25, 55), width=1)
    # Ground line with horizontal glow
    for off, alpha in [(-2, 30), (-1, 60), (0, 130), (1, 60), (2, 30)]:
        draw.line([(0, ground_y + off), (w, ground_y + off)],
                  fill=(50, 50 + alpha // 3, 110 + alpha), width=1)


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

    # Pick action and duration
    action   = _classify_action(hint)
    duration = min(float(max_duration), 30.0)

    # Looping actions get a shorter natural-cycle then we loop the clip
    loop_actions = {"walk", "run", "dance", "fight", "climb", "swim", "sneak", "idle", "wave"}
    if action in loop_actions:
        duration = min(duration, 8.0)  # generate 8 s, then loop to fill

    canvas_w, canvas_h = 720, 540
    ground_y = canvas_h - 60

    times  = np.linspace(0, duration, int(duration * fps))
    frames = []

    print(f"[animation_generator] action='{action}' duration={duration}s frames={len(times)}")

    for t in times:
        pose  = _get_pose(action, float(t), duration)
        frame = _render_frame(pose, canvas_w, canvas_h, ground_y, prompt=hint, t=float(t))
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
    if clip.duration < full_duration and action in loop_actions:
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
