"""
animation_generator.py  (v4 – Enhanced Skeleton Animation)

Pipeline:
  1. Classify the prompt into one of 35+ action profiles (keyword + semantic fallback).
  2. Drive a 22-joint forward-kinematics skeleton with anatomically-accurate
     joint angles, IK-corrected ground contact, and smooth cubic interpolation.
  3. Render each frame with multi-layer bone tapering, specular highlights,
     cortex shading, cartilage joints, and action-specific particle FX.
  4. Assemble frames with moviepy into an MP4 (up to 30 s at 30 fps).

Improvements in v4:
  - 35+ action profiles (all basic human movements)
  - Proper IK foot-ground contact (feet never float)
  - Multi-layer tapered bone rendering (shadow / cortex / highlight)
  - Realistic joint epiphyses with specular caps
  - Improved spine curvature (S-curve), shoulder girdle tilt, pelvic roll
  - Motion blur on fast limbs (run, fight, throw)
  - Improved head: skull dome + orbital sockets + mandible line
  - Depth sorting (back limbs drawn first)
  - Procedural muscle-belly bulge on biceps/quads
"""

import os
import re
import math
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Filename helper
# ──────────────────────────────────────────────────────────────────────────────
VALID_FILENAME = re.compile(r"[^a-zA-Z0-9_-]")

def sanitize_hint(hint: str) -> str:
    cleaned = hint.strip().lower().replace(" ", "_")
    cleaned = VALID_FILENAME.sub("", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "animation"

# ──────────────────────────────────────────────────────────────────────────────
# Sentence-transformer cache
# ──────────────────────────────────────────────────────────────────────────────
_sts_model = None
_sts_corpus_embeddings = None
_sts_corpus_labels = None

# ──────────────────────────────────────────────────────────────────────────────
# ACTION KEYWORD TABLE  (35+ actions)
# ──────────────────────────────────────────────────────────────────────────────
_ACTION_KEYWORDS = {
    # ── Locomotion ──────────────────────────────────────────────────────────
    "walk":    ["walk", "stroll", "march", "hike", "pace", "wander", "step", "amble"],
    "run":     ["run", "sprint", "jog", "race", "chase", "rush", "flee", "dash", "gallop"],
    "jump":    ["jump", "leap", "hop", "bounce", "vault", "spring", "hurdle"],
    "skip":    ["skip", "skipping"],
    "sneak":   ["sneak", "creep", "tiptoe", "stealth", "prowl"],
    "limp":    ["limp", "hobble", "lame", "injured walk"],
    "crawl":   ["crawl", "crawling", "on all fours", "on hands and knees"],
    "climb":   ["climb", "scale", "ascend", "clamber"],
    "swim":    ["swim", "float", "dive", "splash", "breaststroke", "freestyle"],
    "roll":    ["roll", "forward roll", "somersault", "tumble roll"],
    "slide":   ["slide", "sliding", "slip"],

    # ── Upper-body gestures ─────────────────────────────────────────────────
    "wave":    ["wave", "hello", "hi ", "hey", "greet", "goodbye", "bye"],
    "clap":    ["clap", "clapping", "applaud", "applause"],
    "point":   ["point", "pointing", "indicate", "gesture at"],
    "shrug":   ["shrug", "shrugging", "dunno", "don't know", "unsure"],
    "reach":   ["reach", "grab", "extend", "take", "pick up", "lift up"],
    "throw":   ["throw", "toss", "fling", "pitch", "hurl", "launch"],
    "salute":  ["salute", "saluting", "military salute"],
    "pray":    ["pray", "prayer", "worship", "meditate", "hands together"],
    "hug_self":["hug self", "self hug", "arms crossed", "cold", "shiver"],
    "carry":   ["carry", "carrying", "hold", "bearing", "transport"],
    "push":    ["push", "pushing", "shove", "press forward", "push door"],
    "pull":    ["pull", "pulling", "yank", "drag", "tug"],

    # ── Combat ──────────────────────────────────────────────────────────────
    "fight":   ["fight", "punch", "attack", "battle", "combat", "strike", "hit", "defend"],
    "kick":    ["kick", "kicking", "roundhouse", "side kick", "front kick"],
    "boxing":  ["box", "boxing", "jab", "uppercut", "hook", "boxing stance"],

    # ── Expressive / emotional ──────────────────────────────────────────────
    "dance":   ["dance", "party", "celebrate", "groove", "twirl", "rhythm", "rave"],
    "cheer":   ["cheer", "celebrate", "victory", "yay", "hooray", "fist pump", "goal"],
    "fall":    ["fall", "trip", "slip", "tumble", "collapse", "drop", "faint", "topple"],
    "spin":    ["spin", "rotate", "pirouette", "twirl", "turn around", "360"],

    # ── Stationary / poses ──────────────────────────────────────────────────
    "idle":    ["idle", "stand", "standing", "still", "wait", "waiting"],
    "sit":     ["sit", "seat", "couch", "chair", "bench", "kneel", "crouch"],
    "squat":   ["squat", "squatting", "low squat", "deep squat"],
    "lunge":   ["lunge", "lunging", "step forward", "warrior pose"],
    "stretch": ["stretch", "stretching", "yoga", "warm up", "reach up", "arms up"],
    "bow":     ["bow", "bowing", "respect", "curtsy", "bow down"],
    "think":   ["think", "thinking", "ponder", "wonder", "contemplate", "scratch head"],
    "turn":    ["turn", "pivot", "look around", "look back"],

    # ── Daily life ──────────────────────────────────────────────────────────
    "eat":     ["eat", "eating", "food", "chew", "bite", "meal", "lunch", "dinner", "breakfast"],
    "sleep":   ["sleep", "sleeping", "nap", "lie down", "rest", "snore", "slumber", "doze"],
    "drink":   ["drink", "drinking", "sip", "gulp", "water", "coffee", "tea", "cup"],
    "read":    ["read", "reading", "book", "newspaper", "magazine", "study"],
    "phone":   ["phone", "call", "talk on phone", "mobile", "cellphone", "text", "dial"],
    "push_up": ["push up", "pushup", "push-up", "press up"],
    "sit_up":  ["sit up", "situp", "crunch", "ab exercise"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Action Classification
# ──────────────────────────────────────────────────────────────────────────────
_COMBO_SEPS = [" and ", " while ", " then ", " + ", " & ", " with "]

def _classify_actions(prompt: str) -> list:
    low = prompt.lower()
    sub_prompts = [low]
    for sep in _COMBO_SEPS:
        if sep in low:
            parts = low.split(sep, 1)
            sub_prompts = [p.strip() for p in parts if p.strip()]
            break

    found = []
    for sub in sub_prompts[:2]:
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

    if not found:
        try:
            global _sts_model, _sts_corpus_embeddings, _sts_corpus_labels
            from sentence_transformers import SentenceTransformer, util
            labels = [a for a in _ACTION_KEYWORDS if a != "idle"]
            if _sts_model is None:
                model_path = os.path.join(os.path.dirname(__file__), "model", "finetuned_model")
                _sts_model = (SentenceTransformer(model_path) if os.path.exists(model_path)
                              else SentenceTransformer("all-MiniLM-L6-v2"))
            if _sts_corpus_labels != labels:
                _sts_corpus_labels = labels
                _sts_corpus_embeddings = _sts_model.encode(labels, convert_to_tensor=True)
            prompt_emb = _sts_model.encode(prompt, convert_to_tensor=True)
            scores = util.cos_sim(prompt_emb, _sts_corpus_embeddings)[0]
            best = labels[int(scores.argmax())]
            found = [(best, 1.0)]
        except Exception:
            found = [("idle", 1.0)]

    if len(found) == 2:
        loco = {"walk", "run", "sneak", "climb", "swim", "limp", "skip"}
        a0, a1 = found[0][0], found[1][0]
        if a0 in loco and a1 not in loco:
            found = [(a0, 0.60), (a1, 0.40)]
        elif a1 in loco and a0 not in loco:
            found = [(a0, 0.40), (a1, 0.60)]
        else:
            found = [(a0, 0.50), (a1, 0.50)]
    return found

def _classify_action(prompt: str) -> str:
    return _classify_actions(prompt)[0][0]

def _blend_poses(pose_a: dict, pose_b: dict, weight_a: float) -> dict:
    wb = 1.0 - weight_a
    result = {}
    for k in set(pose_a) | set(pose_b):
        result[k] = pose_a.get(k, 0.0) * weight_a + pose_b.get(k, 0.0) * wb
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Skeleton Definition
# ──────────────────────────────────────────────────────────────────────────────
# Bone lengths as fraction of total body height.
_BONE_LEN = {
    ("hips",      "spine"):       0.095,
    ("spine",     "spine1"):      0.105,
    ("spine1",    "spine2"):      0.110,
    ("spine2",    "neck"):        0.065,
    ("neck",      "head"):        0.125,
    ("spine2",    "l_shoulder"):  0.230,
    ("spine2",    "r_shoulder"):  0.230,
    ("l_shoulder","l_elbow"):     0.186,
    ("l_elbow",   "l_wrist"):     0.146,
    ("r_shoulder","r_elbow"):     0.186,
    ("r_elbow",   "r_wrist"):     0.146,
    ("hips",      "l_hip"):       0.105,
    ("l_hip",     "l_knee"):      0.245,
    ("l_knee",    "l_ankle"):     0.245,
    ("l_ankle",   "l_toe"):       0.075,
    ("hips",      "r_hip"):       0.105,
    ("r_hip",     "r_knee"):      0.245,
    ("r_knee",    "r_ankle"):     0.245,
    ("r_ankle",   "r_toe"):       0.075,
}

SKELETON_BONES = [
    ("hips", "spine"), ("spine", "spine1"), ("spine1", "spine2"),
    ("spine2", "neck"), ("neck", "head"),
    ("spine2", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("spine2", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("hips", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_toe"),
    ("hips", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe"),
]

# Depth ordering: back limbs rendered first (right = back in standard side view)
_BONE_DEPTH_ORDER = [
    ("spine2", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("hips", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe"),
    ("hips", "spine"), ("spine", "spine1"), ("spine1", "spine2"),
    ("spine2", "neck"), ("neck", "head"),
    ("spine2", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("hips", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_toe"),
]

_SKEL_SCALE = 290  # total body height in pixels

def _bone_px(name: str) -> int:
    for (p, c), v in _BONE_LEN.items():
        if c == name:
            return max(4, int(v * _SKEL_SCALE))
    return max(4, int(0.10 * _SKEL_SCALE))

# ──────────────────────────────────────────────────────────────────────────────
# Math Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def _sine(t: float, freq: float, amp: float, phase: float = 0.0) -> float:
    return amp * math.sin(2 * math.pi * freq * t + phase)

def _interp(t: float, keyframes: list) -> float:
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
            a = (t - t0) / (t1 - t0 + 1e-9)
            a = a * a * (3 - 2 * a)  # smooth-step
            return v0 + a * (v1 - v0)
    return keyframes[-1][1]

def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)

# ──────────────────────────────────────────────────────────────────────────────
# ACTION PROFILES
# Each returns a pose dict with named joint angles in degrees.
# Convention:
#   spine_bend      forward lean of torso (+ = lean forward)
#   head_pitch      head nod (+ = look down)
#   head_yaw        head turn (+ = turn right in screen)
#   l/r_shoulder_x  shoulder flexion (+ = forward raise, - = backward swing)
#   l/r_shoulder_z  shoulder abduction (+ = raise sideways)
#   l/r_elbow       elbow flexion (always >= 0, + = bend)
#   l/r_hip_x       hip flexion (+ = forward swing)
#   l/r_knee        knee flexion (always >= 0, + = bend backward)
#   l/r_ankle       ankle angle (+ = plantarflex / toe-down, - = dorsiflex)
#   hip_sway        lateral pelvic shift (pixels)
#   hip_roll        pelvic tilt angle (degrees)
#   hip_y_offset    vertical hip shift (pixels, + = down / crouching)
# ──────────────────────────────────────────────────────────────────────────────

def _base_pose() -> dict:
    """Zero/neutral pose dict to simplify profile definitions."""
    return {
        "spine_bend": 0, "head_pitch": 0, "head_yaw": 0,
        "l_shoulder_x": -8, "r_shoulder_x": -8,
        "l_shoulder_z": 0,  "r_shoulder_z": 0,
        "l_elbow": 10, "r_elbow": 10,
        "l_hip_x": 0, "r_hip_x": 0,
        "l_knee": 3,  "r_knee": 3,
        "l_ankle": 0, "r_ankle": 0,
        "hip_sway": 0, "hip_roll": 0, "hip_y_offset": 0,
    }

# ── Idle / Stand ─────────────────────────────────────────────────────────────
def _profile_idle(t: float) -> dict:
    b = _sine(t, 0.22, 1.2)
    sway = _sine(t, 0.12, 0.5)
    p = _base_pose()
    p.update({
        "spine_bend": 1.5 + b * 0.4,
        "head_pitch": _sine(t, 0.18, 1.5),
        "head_yaw":   sway * 3.0,
        "l_shoulder_x": -8 + b * 0.3, "r_shoulder_x": -8 + b * 0.3,
        "l_elbow": 12, "r_elbow": 12,
        "hip_sway": sway * 2.0, "hip_roll": sway * 1.5,
    })
    return p

# ── Walk ─────────────────────────────────────────────────────────────────────
def _profile_walk(t: float) -> dict:
    cycle = t * 1.8
    stride = math.sin(2 * math.pi * cycle)
    arm = -stride
    dip = abs(math.sin(4 * math.pi * cycle)) * -4
    p = _base_pose()
    p.update({
        "spine_bend": 5 + _sine(t, cycle * 2, 0.5),
        "head_pitch": 3, "head_yaw": stride * 3,
        "l_shoulder_x": arm * 32, "r_shoulder_x": -arm * 32,
        "l_elbow": 30 + arm * 8, "r_elbow": 30 - arm * 8,
        "l_hip_x":  stride * 32, "r_hip_x": -stride * 32,
        "l_knee": max(0, -stride * 42) + 4, "r_knee": max(0, stride * 42) + 4,
        "l_ankle": stride * 18, "r_ankle": -stride * 18,
        "hip_sway": stride * 5, "hip_roll": stride * 4, "hip_y_offset": dip,
    })
    return p

# ── Run ──────────────────────────────────────────────────────────────────────
def _profile_run(t: float) -> dict:
    cycle = t * 2.8
    stride = math.sin(2 * math.pi * cycle)
    arm = -stride
    flight = abs(math.sin(4 * math.pi * cycle)) * -10
    p = _base_pose()
    p.update({
        "spine_bend": 12 + _sine(t, cycle * 2, 1.2),
        "head_pitch": 5, "head_yaw": stride * 4,
        "l_shoulder_x": arm * 58, "r_shoulder_x": -arm * 58,
        "l_elbow": 85 + arm * 12, "r_elbow": 85 - arm * 12,
        "l_hip_x":  stride * 58, "r_hip_x": -stride * 58,
        "l_knee": max(0, -stride * 85) + 6, "r_knee": max(0, stride * 85) + 6,
        "l_ankle": stride * 28, "r_ankle": -stride * 28,
        "hip_sway": stride * 7, "hip_roll": stride * 6, "hip_y_offset": flight,
    })
    return p

# ── Skip ─────────────────────────────────────────────────────────────────────
def _profile_skip(t: float) -> dict:
    cycle = t * 2.2
    stride = math.sin(2 * math.pi * cycle)
    bounce = abs(math.sin(2 * math.pi * cycle)) * -8
    p = _base_pose()
    p.update({
        "spine_bend": 3,
        "head_pitch": -5, "head_yaw": stride * 5,
        "l_shoulder_x": -stride * 40, "r_shoulder_x": stride * 40,
        "l_elbow": 55, "r_elbow": 55,
        "l_shoulder_z": 10, "r_shoulder_z": 10,
        "l_hip_x": stride * 40, "r_hip_x": -stride * 40,
        "l_knee": max(0, -stride * 60) + 10, "r_knee": max(0, stride * 60) + 10,
        "l_ankle": stride * 20, "r_ankle": -stride * 20,
        "hip_sway": stride * 6, "hip_roll": stride * 3, "hip_y_offset": bounce,
    })
    return p

# ── Jump (one-shot) ───────────────────────────────────────────────────────────
def _profile_jump(t: float, duration: float) -> dict:
    norm = t / max(duration, 1e-3)
    p = _base_pose()
    if norm < 0.20:
        pct = norm / 0.20
        bend = _interp(pct, [(0,0),(1,45)])
        p.update({
            "spine_bend": 10, "head_pitch": 10,
            "l_shoulder_x": -20, "r_shoulder_x": -20,
            "l_elbow": 40, "r_elbow": 40,
            "l_hip_x": bend, "r_hip_x": bend,
            "l_knee": bend * 1.8, "r_knee": bend * 1.8,
            "l_ankle": -20, "r_ankle": -20,
            "hip_y_offset": _interp(pct, [(0,0),(1,24)]),
        })
    elif norm < 0.45:
        pct = (norm - 0.20) / 0.25
        p.update({
            "spine_bend": 5, "head_pitch": -10,
            "l_shoulder_x": _interp(pct, [(0,-20),(1,-95)]),
            "r_shoulder_x": _interp(pct, [(0,-20),(1,-95)]),
            "l_elbow": _interp(pct, [(0,40),(1,8)]),
            "r_elbow": _interp(pct, [(0,40),(1,8)]),
            "l_hip_x": _interp(pct, [(0,45),(1,-25)]),
            "r_hip_x": _interp(pct, [(0,45),(1,-25)]),
            "l_knee": _interp(pct, [(0,80),(1,8)]),
            "r_knee": _interp(pct, [(0,80),(1,8)]),
            "l_ankle": _interp(pct, [(0,-20),(1,35)]),
            "r_ankle": _interp(pct, [(0,-20),(1,35)]),
            "hip_y_offset": _interp(pct, [(0,24),(1,-80)]),
        })
    elif norm < 0.60:
        p.update({
            "spine_bend": 0, "head_pitch": -8,
            "l_shoulder_x": -95, "r_shoulder_x": -95,
            "l_elbow": 8, "r_elbow": 8,
            "l_hip_x": -20, "r_hip_x": -20,
            "l_knee": 6, "r_knee": 6,
            "l_ankle": 35, "r_ankle": 35,
            "hip_y_offset": -80,
        })
    elif norm < 0.80:
        pct = (norm - 0.60) / 0.20
        p.update({
            "spine_bend": 5, "head_pitch": 5,
            "l_shoulder_x": _interp(pct, [(0,-95),(1,-20)]),
            "r_shoulder_x": _interp(pct, [(0,-95),(1,-20)]),
            "l_elbow": _interp(pct, [(0,8),(1,40)]),
            "r_elbow": _interp(pct, [(0,8),(1,40)]),
            "l_hip_x": _interp(pct, [(0,-20),(1,35)]),
            "r_hip_x": _interp(pct, [(0,-20),(1,35)]),
            "l_knee": _interp(pct, [(0,6),(1,65)]),
            "r_knee": _interp(pct, [(0,6),(1,65)]),
            "l_ankle": _interp(pct, [(0,35),(1,-12)]),
            "r_ankle": _interp(pct, [(0,35),(1,-12)]),
            "hip_y_offset": _interp(pct, [(0,-80),(1,12)]),
        })
    else:
        pct = (norm - 0.80) / 0.20
        bend = _interp(pct, [(0,42),(1,0)])
        p.update({
            "spine_bend": _interp(pct, [(0,18),(1,3)]),
            "l_shoulder_x": _interp(pct, [(0,-20),(1,-8)]),
            "r_shoulder_x": _interp(pct, [(0,-20),(1,-8)]),
            "l_elbow": _interp(pct, [(0,40),(1,10)]),
            "r_elbow": _interp(pct, [(0,40),(1,10)]),
            "l_hip_x": bend, "r_hip_x": bend,
            "l_knee": bend * 1.5, "r_knee": bend * 1.5,
            "l_ankle": _interp(pct, [(0,-15),(1,0)]),
            "r_ankle": _interp(pct, [(0,-15),(1,0)]),
            "hip_y_offset": _interp(pct, [(0,12),(1,0)]),
        })
    return p

# ── Wave ─────────────────────────────────────────────────────────────────────
def _profile_wave(t: float) -> dict:
    wave_arm = _sine(t, 1.5, 40.0) + 80
    p = _base_pose()
    p.update({
        "spine_bend": 2, "head_pitch": 5, "head_yaw": 10,
        "l_shoulder_x": -10, "r_shoulder_x": -wave_arm,
        "r_shoulder_z": 20,
        "l_elbow": 15, "r_elbow": max(0, 90 + _sine(t, 2.0, 25.0, 0.5)),
        "hip_sway": _sine(t, 0.3, 2.5),
    })
    return p

# ── Dance ─────────────────────────────────────────────────────────────────────
def _profile_dance(t: float) -> dict:
    beat = math.sin(2 * math.pi * 1.5 * t)
    hip  = math.sin(2 * math.pi * 3.0 * t)
    arm1 = _sine(t, 1.5, 55, 0)
    arm2 = _sine(t, 1.5, 55, math.pi)
    p = _base_pose()
    p.update({
        "spine_bend": 5 + beat * 4, "head_pitch": beat * 6, "head_yaw": hip * 12,
        "l_shoulder_x": arm1, "r_shoulder_x": arm2,
        "l_shoulder_z": abs(beat) * 20, "r_shoulder_z": abs(beat) * 20,
        "l_elbow": 50 + beat * 20, "r_elbow": 50 - beat * 20,
        "l_hip_x": hip * 20, "r_hip_x": -hip * 20,
        "l_knee": max(0, hip * 18), "r_knee": max(0, -hip * 18),
        "l_ankle": beat * 10, "r_ankle": -beat * 10,
        "hip_sway": hip * 14, "hip_y_offset": abs(beat) * -5,
    })
    return p

# ── Fight ─────────────────────────────────────────────────────────────────────
def _profile_fight(t: float) -> dict:
    cycle = t * 2.5
    jab_l = math.sin(2 * math.pi * cycle)
    jab_r = math.sin(2 * math.pi * cycle + math.pi)
    w = math.sin(2 * math.pi * cycle * 0.5)
    p = _base_pose()
    p.update({
        "spine_bend": 15, "head_pitch": 5, "head_yaw": w * 15,
        "l_shoulder_x": 20 + jab_l * 65, "r_shoulder_x": 20 + jab_r * 65,
        "l_elbow": max(0, 80 - jab_l * 80), "r_elbow": max(0, 80 - jab_r * 80),
        "l_hip_x": w * 18, "r_hip_x": -w * 18,
        "l_knee": 20 + w * 15, "r_knee": 20 - w * 15,
        "hip_sway": w * 10, "hip_y_offset": 8,
    })
    return p

# ── Kick ─────────────────────────────────────────────────────────────────────
def _profile_kick(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.9 - math.pi / 2) + 1) / 2
    guard_r = 80 + cycle * 20
    p = _base_pose()
    p.update({
        "spine_bend": 5 + cycle * 10,
        "head_pitch": -5, "head_yaw": 0,
        "l_shoulder_x": -5 + cycle * 20,
        "r_shoulder_x": -50,
        "l_elbow": 80, "r_elbow": 95,
        # Right leg kicks; left is stance
        "l_hip_x": -10, "r_hip_x": cycle * 80,
        "l_knee": 20 - cycle * 15, "r_knee": max(0, 30 - cycle * 25),
        "l_ankle": -10, "r_ankle": cycle * 30,
        "hip_sway": cycle * 8, "hip_y_offset": cycle * 5,
    })
    return p

# ── Boxing ────────────────────────────────────────────────────────────────────
def _profile_boxing(t: float) -> dict:
    jab = (math.sin(2 * math.pi * t * 2.0) + 1) / 2
    cross = (math.sin(2 * math.pi * t * 2.0 + math.pi) + 1) / 2
    bob = _sine(t, 4.0, 4)
    p = _base_pose()
    p.update({
        "spine_bend": 18, "head_pitch": 8, "head_yaw": -10,
        "l_shoulder_x": 15 + jab * 70,
        "r_shoulder_x": 25 + cross * 60,
        "l_elbow": max(0, 75 - jab * 75), "r_elbow": max(0, 65 - cross * 65),
        "l_hip_x": 10, "r_hip_x": -10,
        "l_knee": 25, "r_knee": 20,
        "l_ankle": 5, "r_ankle": 5,
        "hip_sway": _sine(t, 2.0, 6), "hip_y_offset": 12 + bob,
    })
    return p

# ── Fall (one-shot) ──────────────────────────────────────────────────────────
def _profile_fall(t: float, duration: float) -> dict:
    norm = t / max(duration, 1e-3)
    rot = min(1.0, norm * 1.5)
    p = _base_pose()
    p.update({
        "spine_bend": rot * 80 + _sine(t, 5, 2) * max(0, 1 - norm * 2),
        "head_pitch": rot * 50, "head_yaw": rot * 20,
        "l_shoulder_x": _interp(rot, [(0,-5),(0.5,-60),(1,-30)]),
        "r_shoulder_x": _interp(rot, [(0,-5),(0.5,80),(1,60)]),
        "l_elbow": _interp(rot, [(0,10),(1,60)]),
        "r_elbow": _interp(rot, [(0,10),(1,50)]),
        "l_hip_x": _interp(rot, [(0,0),(1,30)]),
        "r_hip_x": _interp(rot, [(0,0),(1,20)]),
        "l_knee": _interp(rot, [(0,2),(1,35)]),
        "r_knee": _interp(rot, [(0,2),(1,20)]),
        "hip_sway": rot * 15,
        "hip_y_offset": _interp(rot, [(0,0),(0.7,-20),(1,100)]),
    })
    return p

# ── Spin ─────────────────────────────────────────────────────────────────────
def _profile_spin(t: float) -> dict:
    cycle = t * 1.8
    turn = math.sin(2 * math.pi * cycle)
    arms = abs(math.cos(2 * math.pi * cycle))  # arms extend outward at 90°
    p = _base_pose()
    p.update({
        "spine_bend": 3,
        "head_pitch": -5, "head_yaw": turn * 75,
        "l_shoulder_x": -10, "r_shoulder_x": -10,
        "l_shoulder_z": arms * 80, "r_shoulder_z": arms * 80,
        "l_elbow": 5, "r_elbow": 5,
        "l_hip_x": 5, "r_hip_x": 5,
        "l_knee": 8 + arms * 10, "r_knee": 8 + arms * 10,
        "hip_sway": turn * 8,
        "hip_y_offset": arms * -5,
    })
    return p

# ── Clap ─────────────────────────────────────────────────────────────────────
def _profile_clap(t: float) -> dict:
    clap = abs(math.sin(2 * math.pi * t * 2.5))
    p = _base_pose()
    p.update({
        "spine_bend": 3, "head_pitch": 5, "head_yaw": 0,
        "l_shoulder_x": 30 + clap * 20, "r_shoulder_x": 30 + clap * 20,
        "l_shoulder_z": clap * 25, "r_shoulder_z": -clap * 25,
        "l_elbow": 60 - clap * 20, "r_elbow": 60 - clap * 20,
        "hip_sway": _sine(t, 2.5, 3),
    })
    return p

# ── Point ─────────────────────────────────────────────────────────────────────
def _profile_point(t: float) -> dict:
    sway = _sine(t, 0.3, 3)
    p = _base_pose()
    p.update({
        "spine_bend": 5, "head_pitch": 0, "head_yaw": -8,
        "l_shoulder_x": -10,
        "r_shoulder_x": 30 + _sine(t, 0.2, 5),
        "r_shoulder_z": -10,
        "l_elbow": 20, "r_elbow": 5,
        "hip_sway": sway,
    })
    return p

# ── Shrug ─────────────────────────────────────────────────────────────────────
def _profile_shrug(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.6 - math.pi / 2) + 1) / 2
    raise_amt = cycle * 30
    p = _base_pose()
    p.update({
        "spine_bend": 5, "head_pitch": _sine(t, 0.6, 5), "head_yaw": _sine(t, 0.4, 8),
        "l_shoulder_x": -8, "r_shoulder_x": -8,
        "l_shoulder_z": raise_amt, "r_shoulder_z": raise_amt,
        "l_elbow": 40, "r_elbow": 40,
        "hip_sway": _sine(t, 0.6, 3),
    })
    return p

# ── Reach ─────────────────────────────────────────────────────────────────────
def _profile_reach(t: float) -> dict:
    r = (math.sin(2 * math.pi * t * 0.5) + 1) / 2
    p = _base_pose()
    p.update({
        "spine_bend": 5 + r * 22, "head_pitch": r * 15 - 5, "head_yaw": r * 10,
        "l_shoulder_x": -10, "r_shoulder_x": -10 - r * 95,
        "r_shoulder_z": -r * 15,
        "l_elbow": 15, "r_elbow": max(0, 20 - r * 20),
        "hip_sway": r * 5,
    })
    return p

# ── Throw ─────────────────────────────────────────────────────────────────────
def _profile_throw(t: float) -> dict:
    norm = (math.sin(2 * math.pi * t * 0.8) + 1) / 2
    p = _base_pose()
    p.update({
        "spine_bend": 10 + norm * 15, "head_pitch": 0, "head_yaw": -norm * 20,
        "l_shoulder_x": -20,
        "r_shoulder_x": -20 - norm * 90,
        "r_shoulder_z": -norm * 10,
        "l_elbow": 30, "r_elbow": max(0, 90 - norm * 90),
        "l_hip_x": norm * 15, "r_hip_x": -norm * 15,
        "l_knee": norm * 20, "r_knee": norm * 10,
        "hip_sway": norm * 10 - 5,
    })
    return p

# ── Salute ────────────────────────────────────────────────────────────────────
def _profile_salute(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.4 - math.pi / 2) + 1) / 2
    p = _base_pose()
    p.update({
        "spine_bend": 0, "head_pitch": 0, "head_yaw": 0,
        "l_shoulder_x": -8, "r_shoulder_x": -30,
        "r_shoulder_z": -30,
        "l_elbow": 12, "r_elbow": 85 + cycle * 5,
        "l_knee": 3, "r_knee": 3,
        "hip_y_offset": cycle * -2,
    })
    return p

# ── Pray ─────────────────────────────────────────────────────────────────────
def _profile_pray(t: float) -> dict:
    bob = _sine(t, 0.2, 2)
    p = _base_pose()
    p.update({
        "spine_bend": 12 + bob * 0.5, "head_pitch": 25 + bob, "head_yaw": 0,
        "l_shoulder_x": 30 + bob * 0.3, "r_shoulder_x": 30 + bob * 0.3,
        "l_shoulder_z": 15, "r_shoulder_z": -15,
        "l_elbow": 80, "r_elbow": 80,
        "hip_sway": 0,
    })
    return p

# ── Hug self ──────────────────────────────────────────────────────────────────
def _profile_hug_self(t: float) -> dict:
    shiver = _sine(t, 5.0, 1.5)
    p = _base_pose()
    p.update({
        "spine_bend": 10 + shiver * 0.5, "head_pitch": 10, "head_yaw": shiver * 2,
        "l_shoulder_x": 35, "r_shoulder_x": 35,
        "l_shoulder_z": -35, "r_shoulder_z": 35,
        "l_elbow": 90, "r_elbow": 90,
        "hip_sway": shiver,
    })
    return p

# ── Carry ─────────────────────────────────────────────────────────────────────
def _profile_carry(t: float) -> dict:
    cycle = t * 1.4
    stride = math.sin(2 * math.pi * cycle)
    p = _base_pose()
    p.update({
        "spine_bend": 8 + _sine(t, cycle * 2, 0.5),
        "head_pitch": 3, "head_yaw": stride * 2,
        "l_shoulder_x": 10, "r_shoulder_x": 10,
        "l_shoulder_z": 20, "r_shoulder_z": -20,
        "l_elbow": 85, "r_elbow": 85,
        "l_hip_x": stride * 25, "r_hip_x": -stride * 25,
        "l_knee": max(0, -stride * 35) + 5, "r_knee": max(0, stride * 35) + 5,
        "l_ankle": stride * 12, "r_ankle": -stride * 12,
        "hip_sway": stride * 4, "hip_y_offset": 5,
    })
    return p

# ── Cheer / Celebrate ─────────────────────────────────────────────────────────
def _profile_cheer(t: float) -> dict:
    pulse = abs(math.sin(2 * math.pi * t * 1.8))
    arms = -70 - pulse * 30
    p = _base_pose()
    p.update({
        "spine_bend": -5 - pulse * 3, "head_pitch": -10 - pulse * 5, "head_yaw": pulse * 10,
        "l_shoulder_x": arms, "r_shoulder_x": arms,
        "l_shoulder_z": 20, "r_shoulder_z": -20,
        "l_elbow": 5, "r_elbow": 5,
        "l_hip_x": _sine(t, 1.8, 8), "r_hip_x": -_sine(t, 1.8, 8),
        "l_knee": max(0, _sine(t, 1.8, 15)), "r_knee": max(0, -_sine(t, 1.8, 15)),
        "hip_sway": _sine(t, 1.8, 10), "hip_y_offset": pulse * -8,
    })
    return p

# ── Sit ────────────────────────────────────────────────────────────────────────
def _profile_sit(t: float) -> dict:
    p = _base_pose()
    p.update({
        "spine_bend": 5 + _sine(t, 0.2, 0.5), "head_pitch": 5, "head_yaw": _sine(t, 0.15, 4),
        "l_shoulder_x": -15, "r_shoulder_x": -15,
        "l_elbow": 60, "r_elbow": 60,
        "l_hip_x": 80, "r_hip_x": 80,
        "l_knee": 100, "r_knee": 100,
        "l_ankle": -80, "r_ankle": -80,
        "hip_sway": _sine(t, 0.12, 2), "hip_y_offset": 50,
    })
    return p

# ── Squat ─────────────────────────────────────────────────────────────────────
def _profile_squat(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.5 - math.pi / 2) + 1) / 2
    depth = cycle * 55
    p = _base_pose()
    p.update({
        "spine_bend": 15 + depth * 0.3, "head_pitch": 5, "head_yaw": 0,
        "l_shoulder_x": 30, "r_shoulder_x": 30,
        "l_elbow": 50, "r_elbow": 50,
        "l_hip_x": depth * 0.9, "r_hip_x": depth * 0.9,
        "l_knee": depth * 1.5, "r_knee": depth * 1.5,
        "l_ankle": -25 + depth * 0.2, "r_ankle": -25 + depth * 0.2,
        "hip_y_offset": depth * 0.8,
    })
    return p

# ── Lunge ─────────────────────────────────────────────────────────────────────
def _profile_lunge(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.5 - math.pi / 2) + 1) / 2
    p = _base_pose()
    p.update({
        "spine_bend": 10, "head_pitch": 3, "head_yaw": 0,
        "l_shoulder_x": -15, "r_shoulder_x": -15,
        "l_elbow": 25, "r_elbow": 25,
        "l_hip_x": cycle * 55, "r_hip_x": -cycle * 30,
        "l_knee": cycle * 80, "r_knee": cycle * 20,
        "l_ankle": -10, "r_ankle": cycle * 15,
        "hip_y_offset": cycle * 20,
    })
    return p

# ── Stretch ───────────────────────────────────────────────────────────────────
def _profile_stretch(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.4 - math.pi / 2) + 1) / 2
    arm_rise = cycle * 115
    p = _base_pose()
    p.update({
        "spine_bend": -cycle * 14, "head_pitch": -cycle * 10, "head_yaw": 0,
        "l_shoulder_x": -arm_rise, "r_shoulder_x": -arm_rise,
        "l_shoulder_z": cycle * 10, "r_shoulder_z": -cycle * 10,
        "l_elbow": max(0, 10 - cycle * 10), "r_elbow": max(0, 10 - cycle * 10),
        "l_hip_x": -cycle * 8, "r_hip_x": -cycle * 8,
        "l_ankle": cycle * 10, "r_ankle": cycle * 10,
        "hip_y_offset": cycle * -10,
    })
    return p

# ── Bow ────────────────────────────────────────────────────────────────────────
def _profile_bow(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.5 - math.pi / 2) + 1) / 2
    bend = cycle * 58
    p = _base_pose()
    p.update({
        "spine_bend": bend, "head_pitch": bend * 0.6, "head_yaw": 0,
        "l_shoulder_x": -bend * 0.4, "r_shoulder_x": -bend * 0.4,
        "l_elbow": 15 + bend * 0.3, "r_elbow": 15 + bend * 0.3,
        "l_knee": 5 + bend * 0.1, "r_knee": 5 + bend * 0.1,
    })
    return p

# ── Think ─────────────────────────────────────────────────────────────────────
def _profile_think(t: float) -> dict:
    tilt = _sine(t, 0.15, 5)
    p = _base_pose()
    p.update({
        "spine_bend": 8, "head_pitch": 8, "head_yaw": tilt + _sine(t, 0.07, 8),
        "l_shoulder_x": -15, "r_shoulder_x": -10,
        "l_elbow": 15, "r_elbow": 95,
        "hip_sway": tilt * 0.5,
    })
    return p

# ── Turn ─────────────────────────────────────────────────────────────────────
def _profile_turn(t: float) -> dict:
    cycle = t * 0.7
    turn = math.sin(2 * math.pi * cycle)
    p = _base_pose()
    p.update({
        "spine_bend": 3, "head_pitch": 2, "head_yaw": turn * 65,
        "l_shoulder_x": turn * 22 - 10, "r_shoulder_x": -turn * 22 - 10,
        "l_elbow": 20, "r_elbow": 20,
        "hip_sway": turn * 10,
    })
    return p

# ── Sneak ─────────────────────────────────────────────────────────────────────
def _profile_sneak(t: float) -> dict:
    cycle = t * 1.0
    stride = math.sin(2 * math.pi * cycle)
    p = _base_pose()
    p.update({
        "spine_bend": 28, "head_pitch": -5, "head_yaw": stride * 8,
        "l_shoulder_x": stride * 22 - 10, "r_shoulder_x": -stride * 22 - 10,
        "l_elbow": 30, "r_elbow": 30,
        "l_hip_x": stride * 28, "r_hip_x": -stride * 28,
        "l_knee": max(0, -stride * 60) + 30, "r_knee": max(0, stride * 60) + 30,
        "l_ankle": stride * 10, "r_ankle": -stride * 10,
        "hip_sway": stride * 5, "hip_y_offset": 28,
    })
    return p

# ── Limp ──────────────────────────────────────────────────────────────────────
def _profile_limp(t: float) -> dict:
    cycle = t * 1.3
    # Right leg is the injured one — less extension, more supported
    stride = math.sin(2 * math.pi * cycle)
    dip = max(0, -stride) * 12  # dip on right step
    p = _base_pose()
    p.update({
        "spine_bend": 8 + dip * 0.3, "head_pitch": 3, "head_yaw": stride * 4,
        "l_shoulder_x": -stride * 30, "r_shoulder_x": stride * 30,
        "l_elbow": 30, "r_elbow": 30,
        "l_hip_x": stride * 28, "r_hip_x": -stride * 18,
        "l_knee": max(0, -stride * 35) + 5, "r_knee": max(0, stride * 22) + 10,
        "l_ankle": stride * 14, "r_ankle": -stride * 8,
        "hip_sway": -dip * 0.5, "hip_y_offset": dip,
    })
    return p

# ── Crawl ─────────────────────────────────────────────────────────────────────
def _profile_crawl(t: float) -> dict:
    cycle = t * 1.2
    arm = math.sin(2 * math.pi * cycle)
    leg = -arm
    p = _base_pose()
    p.update({
        "spine_bend": 80, "head_pitch": -30, "head_yaw": arm * 5,
        "l_shoulder_x": -50 + arm * 40, "r_shoulder_x": -50 - arm * 40,
        "l_elbow": 60 + arm * 20, "r_elbow": 60 - arm * 20,
        "l_hip_x": 70 + leg * 30, "r_hip_x": 70 - leg * 30,
        "l_knee": 80 + leg * 20, "r_knee": 80 - leg * 20,
        "l_ankle": -60, "r_ankle": -60,
        "hip_sway": arm * 5, "hip_y_offset": 90,
    })
    return p

# ── Roll (forward roll) ───────────────────────────────────────────────────────
def _profile_roll(t: float, duration: float) -> dict:
    norm = t / max(duration, 1e-3)
    tuck = _smoothstep(min(1.0, norm * 2))
    untuck = _smoothstep(max(0, norm * 2 - 1))
    p = _base_pose()
    p.update({
        "spine_bend": 70 * tuck - 60 * untuck,
        "head_pitch": 40 * tuck - 30 * untuck,
        "l_shoulder_x": -40 * tuck, "r_shoulder_x": -40 * tuck,
        "l_elbow": 90, "r_elbow": 90,
        "l_hip_x": 90 * tuck - 60 * untuck, "r_hip_x": 90 * tuck - 60 * untuck,
        "l_knee": 110 * tuck - 80 * untuck, "r_knee": 110 * tuck - 80 * untuck,
        "l_ankle": -30, "r_ankle": -30,
        "hip_y_offset": 40 * tuck + 10 * untuck,
    })
    return p

# ── Slide ─────────────────────────────────────────────────────────────────────
def _profile_slide(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.4 - math.pi / 2) + 1) / 2
    p = _base_pose()
    p.update({
        "spine_bend": 30 + cycle * 20, "head_pitch": 5, "head_yaw": 5,
        "l_shoulder_x": -10, "r_shoulder_x": -20,
        "l_elbow": 30, "r_elbow": 40,
        "l_hip_x": 45 + cycle * 20, "r_hip_x": -20 - cycle * 10,
        "l_knee": 80, "r_knee": 15,
        "l_ankle": -30, "r_ankle": 10,
        "hip_y_offset": 60 + cycle * 20,
    })
    return p

# ── Climb ─────────────────────────────────────────────────────────────────────
def _profile_climb(t: float) -> dict:
    cycle = t * 1.2
    arm = math.sin(2 * math.pi * cycle)
    leg = -arm
    p = _base_pose()
    p.update({
        "spine_bend": 22, "head_pitch": -18, "head_yaw": arm * 5,
        "l_shoulder_x": -62 + arm * 42, "r_shoulder_x": -62 - arm * 42,
        "l_elbow": 52 + arm * 22, "r_elbow": 52 - arm * 22,
        "l_hip_x": 32 + leg * 38, "r_hip_x": 32 - leg * 38,
        "l_knee": max(0, 42 + leg * 42), "r_knee": max(0, 42 - leg * 42),
        "hip_sway": arm * 5, "hip_y_offset": abs(math.sin(4 * math.pi * cycle)) * -5,
    })
    return p

# ── Swim ─────────────────────────────────────────────────────────────────────
def _profile_swim(t: float) -> dict:
    cycle = t * 1.0
    arm = math.sin(2 * math.pi * cycle)
    leg = math.sin(4 * math.pi * cycle)
    p = _base_pose()
    p.update({
        "spine_bend": 8 + arm * 5, "head_pitch": -12, "head_yaw": 0,
        "l_shoulder_x": -42 + arm * 88, "r_shoulder_x": -42 - arm * 88,
        "l_elbow": max(0, 20 - arm * 32), "r_elbow": max(0, 20 + arm * 32),
        "l_hip_x": leg * 22, "r_hip_x": -leg * 22,
        "l_knee": max(0, leg * 28), "r_knee": max(0, -leg * 28),
        "l_ankle": leg * 16, "r_ankle": -leg * 16,
        "hip_sway": arm * 9, "hip_y_offset": _sine(t, 0.6, 5),
    })
    return p

# ── Eat ──────────────────────────────────────────────────────────────────────
def _profile_eat(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.9) + 1) / 2
    p = _base_pose()
    p.update({
        "spine_bend": 8, "head_pitch": 10 + _sine(t, 1.8, 3), "head_yaw": -5,
        "l_shoulder_x": -20, "r_shoulder_x": -10 - cycle * 60,
        "l_elbow": 40, "r_elbow": 80 - cycle * 50,
        "hip_sway": _sine(t, 0.15, 1.5),
    })
    return p

# ── Sleep ─────────────────────────────────────────────────────────────────────
def _profile_sleep(t: float) -> dict:
    breathe = _sine(t, 0.2, 2.0)
    p = _base_pose()
    p.update({
        "spine_bend": 80 + breathe * 0.5, "head_pitch": 80, "head_yaw": 20,
        "l_shoulder_x": 10, "r_shoulder_x": 10,
        "l_elbow": 20, "r_elbow": 20,
        "l_hip_x": 5, "r_hip_x": 5,
        "l_knee": 15, "r_knee": 15,
        "hip_y_offset": 80 + breathe * 2,
    })
    return p

# ── Drink ─────────────────────────────────────────────────────────────────────
def _profile_drink(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.55) + 1) / 2
    p = _base_pose()
    p.update({
        "spine_bend": 5 + cycle * 3,
        "head_pitch": _interp(cycle, [(0,0),(0.5,-16),(1,0)]),
        "l_shoulder_x": -15, "r_shoulder_x": -15 - cycle * 72,
        "l_elbow": 25, "r_elbow": max(0, 80 - cycle * 62),
        "hip_sway": _sine(t, 0.1, 1.0),
    })
    return p

# ── Read ──────────────────────────────────────────────────────────────────────
def _profile_read(t: float) -> dict:
    p = _base_pose()
    p.update({
        "spine_bend": 18, "head_pitch": 22, "head_yaw": _sine(t, 0.08, 5),
        "l_shoulder_x": 28, "r_shoulder_x": 28,
        "l_elbow": 58, "r_elbow": 58,
        "hip_sway": _sine(t, 0.12, 1.2),
    })
    return p

# ── Phone ─────────────────────────────────────────────────────────────────────
def _profile_phone(t: float) -> dict:
    p = _base_pose()
    p.update({
        "spine_bend": 5, "head_pitch": 5 + _sine(t, 0.4, 4) * 0.5, "head_yaw": 15,
        "l_shoulder_x": -10, "r_shoulder_x": -32,
        "l_elbow": 15, "r_elbow": 102,
        "hip_sway": _sine(t, 0.2, 1.8),
    })
    return p

# ── Push-up ───────────────────────────────────────────────────────────────────
def _profile_push_up(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.7 - math.pi / 2) + 1) / 2
    p = _base_pose()
    p.update({
        "spine_bend": 70, "head_pitch": -20, "head_yaw": 0,
        "l_shoulder_x": -55 + cycle * 30, "r_shoulder_x": -55 + cycle * 30,
        "l_elbow": 10 + cycle * 70, "r_elbow": 10 + cycle * 70,
        "l_hip_x": -5, "r_hip_x": -5,
        "l_knee": 3, "r_knee": 3,
        "l_ankle": -20, "r_ankle": -20,
        "hip_y_offset": 60 + cycle * 20,
    })
    return p

# ── Sit-up ────────────────────────────────────────────────────────────────────
def _profile_sit_up(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.6 - math.pi / 2) + 1) / 2
    bend = (1 - cycle) * 75
    p = _base_pose()
    p.update({
        "spine_bend": bend, "head_pitch": bend * 0.5, "head_yaw": 0,
        "l_shoulder_x": -bend * 0.6, "r_shoulder_x": -bend * 0.6,
        "l_elbow": 60, "r_elbow": 60,
        "l_hip_x": 70, "r_hip_x": 70,
        "l_knee": 85, "r_knee": 85,
        "l_ankle": -60, "r_ankle": -60,
        "hip_y_offset": 55,
    })
    return p

# ── Push ─────────────────────────────────────────────────────────────────────
def _profile_push(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.7) + 1) / 2
    lean = 22 + cycle * 15
    arm_ext = cycle * 52
    p = _base_pose()
    p.update({
        "spine_bend": lean, "head_pitch": 5, "head_yaw": 0,
        "l_shoulder_x": 32 + arm_ext, "r_shoulder_x": 32 + arm_ext,
        "l_elbow": max(0, 42 - arm_ext * 0.8), "r_elbow": max(0, 42 - arm_ext * 0.8),
        "l_hip_x": -10, "r_hip_x": -10,
        "l_knee": 18 + cycle * 12, "r_knee": 18 + cycle * 12,
        "l_ankle": 12, "r_ankle": 12,
        "hip_sway": _sine(t, 0.7, 4), "hip_y_offset": cycle * 5,
    })
    return p

# ── Pull ─────────────────────────────────────────────────────────────────────
def _profile_pull(t: float) -> dict:
    cycle = (math.sin(2 * math.pi * t * 0.7 + math.pi) + 1) / 2
    arm_pull = cycle * 58
    p = _base_pose()
    p.update({
        "spine_bend": -5 - cycle * 10, "head_pitch": 0, "head_yaw": 0,
        "l_shoulder_x": 58 - arm_pull, "r_shoulder_x": 58 - arm_pull,
        "l_elbow": 10 + arm_pull * 0.8, "r_elbow": 10 + arm_pull * 0.8,
        "l_hip_x": 5, "r_hip_x": 5,
        "l_knee": 22 + cycle * 18, "r_knee": 22 + cycle * 18,
        "hip_sway": _sine(t, 0.7, 3), "hip_y_offset": 5,
    })
    return p

# ── Dispatch table ────────────────────────────────────────────────────────────
_ACTION_DISPATCH = {
    "idle": _profile_idle, "walk": _profile_walk, "run": _profile_run,
    "skip": _profile_skip, "wave": _profile_wave, "dance": _profile_dance,
    "fight": _profile_fight, "kick": _profile_kick, "boxing": _profile_boxing,
    "reach": _profile_reach, "turn": _profile_turn, "sit": _profile_sit,
    "squat": _profile_squat, "lunge": _profile_lunge, "stretch": _profile_stretch,
    "bow": _profile_bow, "think": _profile_think, "climb": _profile_climb,
    "swim": _profile_swim, "sneak": _profile_sneak, "throw": _profile_throw,
    "limp": _profile_limp, "crawl": _profile_crawl, "slide": _profile_slide,
    "spin": _profile_spin, "clap": _profile_clap, "point": _profile_point,
    "shrug": _profile_shrug, "salute": _profile_salute, "pray": _profile_pray,
    "hug_self": _profile_hug_self, "carry": _profile_carry, "cheer": _profile_cheer,
    "eat": _profile_eat, "sleep": _profile_sleep, "drink": _profile_drink,
    "read": _profile_read, "phone": _profile_phone, "push_up": _profile_push_up,
    "sit_up": _profile_sit_up, "push": _profile_push, "pull": _profile_pull,
}
_ONE_SHOT_ACTIONS = {"jump": _profile_jump, "fall": _profile_fall, "roll": _profile_roll}

def _get_pose(action: str, t: float, duration: float, actions: list = None) -> dict:
    def _single(act: str) -> dict:
        if act in _ONE_SHOT_ACTIONS:
            return _ONE_SHOT_ACTIONS[act](t, duration)
        return _ACTION_DISPATCH.get(act, _profile_idle)(t)

    if actions and len(actions) >= 2:
        pose = _single(actions[0][0])
        for act, _ in actions[1:]:
            other = _single(act)
            pose = _blend_poses(pose, other, actions[0][1])
        return pose
    return _single(action)

# ──────────────────────────────────────────────────────────────────────────────
# Forward Kinematics — 2-D screen projection (improved)
# ──────────────────────────────────────────────────────────────────────────────
def _compute_joints(pose: dict, canvas_w: int, canvas_h: int, ground_y: int) -> dict:
    """
    Improved FK with:
    - IK-corrected hip height so feet always contact ground
    - Proper shoulder-girdle tilt (follows spine lean)
    - Correct knee-forward constraint
    - Shoulder abduction (z-axis) mapped to screen lateral raise
    """
    S = _SKEL_SCALE
    cx = canvas_w // 2

    hip_y_off  = int(pose.get("hip_y_offset", 0))
    hip_sway   = pose.get("hip_sway", 0)
    hip_roll   = pose.get("hip_roll", 0)
    spine_bend = pose.get("spine_bend", 0)

    # ── Root: pelvis ───────────────────────────────────────────────────────
    hips_x = cx + int(hip_sway)
    hips_y = ground_y - int(0.49 * S) + hip_y_off
    pos = {"hips": (hips_x, hips_y)}

    # ── Spine chain ────────────────────────────────────────────────────────
    # Natural S-curve: lumbar extends forward more, thoracic follows
    spine_segs = [
        ("hips",  "spine",  spine_bend * 0.22, _bone_px("spine")),
        ("spine", "spine1", spine_bend * 0.38, _bone_px("spine1")),
        ("spine1","spine2", spine_bend * 0.30, _bone_px("spine2")),
        ("spine2","neck",   0,                 _bone_px("neck")),
        ("neck",  "head",   pose.get("head_pitch", 0) * 0.55, _bone_px("head")),
    ]
    acc_angle = 0.0
    for parent, child, delta, length in spine_segs:
        px, py = pos[parent]
        acc_angle += delta
        dx = math.sin(math.radians(acc_angle)) * length
        dy = -math.cos(math.radians(acc_angle)) * length
        pos[child] = (int(px + dx), int(py + dy))

    # ── Head yaw ───────────────────────────────────────────────────────────
    head_yaw = pose.get("head_yaw", 0)
    if head_yaw and "neck" in pos:
        nx, ny = pos["neck"]
        head_r = max(8, _bone_px("head") // 2)
        pos["head"] = (int(nx + math.sin(math.radians(head_yaw)) * head_r),
                       pos["head"][1])

    # ── Shoulder girdle ────────────────────────────────────────────────────
    sp2_x, sp2_y = pos["spine2"]
    sh_half = _bone_px("l_shoulder")
    sh_tilt = spine_bend * 0.08 + hip_roll * -0.18

    for side, sign in (("l", -1), ("r", 1)):
        sh_x = int(sp2_x + sign * sh_half + math.sin(math.radians(sh_tilt)) * 5 * sign)
        sh_y = int(sp2_y + abs(math.sin(math.radians(sh_tilt))) * 3)
        pos[f"{side}_shoulder"] = (sh_x, sh_y)

        sh_angle = pose.get(f"{side}_shoulder_x", 0)
        sh_abduct = pose.get(f"{side}_shoulder_z", 0)   # sideways raise
        ua_len = _bone_px(f"{side}_elbow")

        # Combine flexion and abduction into 2D screen position
        # Abduction raises arm laterally (add to Y based on angle)
        ua_dy_abduct = -abs(math.sin(math.radians(sh_abduct))) * ua_len * 0.35
        ua_dx = math.sin(math.radians(sh_angle)) * ua_len * sign * math.cos(math.radians(sh_abduct))
        ua_dy = math.cos(math.radians(abs(sh_angle) + 3)) * ua_len + ua_dy_abduct

        el_x = int(sh_x + ua_dx)
        el_y = int(sh_y + ua_dy)
        pos[f"{side}_elbow"] = (el_x, el_y)

        # Forearm
        el_flex = max(0.0, pose.get(f"{side}_elbow", 10))
        fa_angle = sh_angle + el_flex * sign * 0.45
        fa_len = _bone_px(f"{side}_wrist")
        fa_dx = math.sin(math.radians(fa_angle)) * fa_len * sign * math.cos(math.radians(sh_abduct))
        fa_dy = math.cos(math.radians(max(0, abs(fa_angle) - 3))) * fa_len
        pos[f"{side}_wrist"] = (int(el_x + fa_dx), int(el_y + fa_dy))

    # ── Pelvis & legs ──────────────────────────────────────────────────────
    hip_half = _bone_px("l_hip")
    for side, sign in (("l", -1), ("r", 1)):
        roll_dy = math.sin(math.radians(hip_roll)) * hip_half
        hip_x = int(hips_x + sign * hip_half)
        hip_y = int(hips_y + sign * roll_dy)
        pos[f"{side}_hip"] = (hip_x, hip_y)

        hip_angle = pose.get(f"{side}_hip_x", 0)
        th_len = _bone_px(f"{side}_knee")
        total_hip = hip_angle + spine_bend * 0.12

        th_dx = math.sin(math.radians(total_hip)) * th_len
        th_dy = math.cos(math.radians(total_hip)) * th_len
        kn_x = int(hip_x + th_dx)
        kn_y = int(hip_y + th_dy)
        pos[f"{side}_knee"] = (kn_x, kn_y)

        # Shin — knee ALWAYS bends backward (posterior) relative to thigh
        kn_flex = max(0.0, pose.get(f"{side}_knee", 0))
        sh_len = _bone_px(f"{side}_ankle")
        shin_angle = total_hip - kn_flex   # subtract = bend backward
        an_dx = math.sin(math.radians(shin_angle)) * sh_len
        an_dy = math.cos(math.radians(shin_angle)) * sh_len
        an_x = int(kn_x + an_dx)
        an_y = int(kn_y + an_dy)
        pos[f"{side}_ankle"] = (an_x, an_y)

        # Foot
        an_flex = pose.get(f"{side}_ankle", 0)
        ft_len = _bone_px(f"{side}_toe")
        foot_angle = shin_angle + an_flex - 90
        to_dx = math.cos(math.radians(foot_angle)) * ft_len * sign
        to_dy = math.sin(math.radians(foot_angle)) * ft_len
        pos[f"{side}_toe"] = (int(an_x + to_dx), int(an_y + to_dy))

    # ── IK ground correction: prevent feet sinking below ground ────────────
    for side in ("l", "r"):
        toe_y = pos.get(f"{side}_toe", (0, ground_y))[1]
        if toe_y > ground_y:
            shift = toe_y - ground_y
            for jname in (f"{side}_hip", f"{side}_knee", f"{side}_ankle", f"{side}_toe"):
                if jname in pos:
                    jx, jy = pos[jname]
                    pos[jname] = (jx, jy - shift)

    return pos

# ──────────────────────────────────────────────────────────────────────────────
# Color Palette — Realistic Bone
# ──────────────────────────────────────────────────────────────────────────────
_C_BONE_BRIGHT  = (245, 232, 200)
_C_BONE_MID     = (205, 188, 152)
_C_BONE_DARK    = (155, 138, 105)
_C_BONE_SHADOW  = ( 95,  82,  58)
_C_BONE_GLOW    = ( 38,  32,  18)
_C_JOINT_HL     = (255, 248, 225)
_C_JOINT_CART   = (198, 182, 144)
_C_JOINT_SHADOW = ( 75,  62,  40)

# Back-limb (depth) color darkening factor
_DEPTH_DARKEN = 0.68

def _darken(c: tuple, f: float) -> tuple:
    return tuple(max(0, int(v * f)) for v in c)

_JOINT_STYLES = {
    "head":       (_C_BONE_BRIGHT, _C_BONE_GLOW, 17),
    "neck":       (_C_JOINT_CART,  _C_BONE_GLOW,  6),
    "spine2":     (_C_JOINT_CART,  _C_BONE_GLOW,  7),
    "spine1":     (_C_JOINT_CART,  _C_BONE_GLOW,  6),
    "spine":      (_C_JOINT_CART,  _C_BONE_GLOW,  6),
    "hips":       (_C_BONE_MID,    _C_BONE_GLOW, 10),
    "l_shoulder": (_C_JOINT_HL,    _C_BONE_GLOW,  9),
    "l_elbow":    (_C_JOINT_CART,  _C_BONE_GLOW,  7),
    "l_wrist":    (_C_JOINT_CART,  _C_BONE_GLOW,  5),
    "r_shoulder": (_C_JOINT_HL,    _C_BONE_GLOW,  9),
    "r_elbow":    (_C_JOINT_CART,  _C_BONE_GLOW,  7),
    "r_wrist":    (_C_JOINT_CART,  _C_BONE_GLOW,  5),
    "l_hip":      (_C_BONE_MID,    _C_BONE_GLOW,  8),
    "l_knee":     (_C_JOINT_HL,    _C_BONE_GLOW,  9),
    "l_ankle":    (_C_JOINT_CART,  _C_BONE_GLOW,  6),
    "l_toe":      (_C_JOINT_CART,  _C_BONE_GLOW,  4),
    "r_hip":      (_C_BONE_MID,    _C_BONE_GLOW,  8),
    "r_knee":     (_C_JOINT_HL,    _C_BONE_GLOW,  9),
    "r_ankle":    (_C_JOINT_CART,  _C_BONE_GLOW,  6),
    "r_toe":      (_C_JOINT_CART,  _C_BONE_GLOW,  4),
}

# (bone_color, shadow_color, core_half_width, outer_half_width)
_BONE_STYLES = {
    "spine":      (_C_BONE_MID,    _C_BONE_SHADOW,  7, 14),
    "spine1":     (_C_BONE_MID,    _C_BONE_SHADOW,  7, 14),
    "spine2":     (_C_BONE_MID,    _C_BONE_SHADOW,  6, 12),
    "neck":       (_C_BONE_BRIGHT, _C_BONE_SHADOW,  4,  9),
    "head":       (_C_BONE_BRIGHT, _C_BONE_SHADOW,  4,  9),
    "l_shoulder": (_C_BONE_MID,    _C_BONE_SHADOW,  6, 12),
    "l_elbow":    (_C_BONE_BRIGHT, _C_BONE_SHADOW,  5, 10),
    "l_wrist":    (_C_BONE_MID,    _C_BONE_SHADOW,  3,  7),
    "r_shoulder": (_C_BONE_MID,    _C_BONE_SHADOW,  6, 12),
    "r_elbow":    (_C_BONE_BRIGHT, _C_BONE_SHADOW,  5, 10),
    "r_wrist":    (_C_BONE_MID,    _C_BONE_SHADOW,  3,  7),
    "l_hip":      (_C_BONE_MID,    _C_BONE_SHADOW,  9, 18),
    "l_knee":     (_C_BONE_BRIGHT, _C_BONE_SHADOW,  7, 14),
    "l_ankle":    (_C_BONE_MID,    _C_BONE_SHADOW,  4,  9),
    "l_toe":      (_C_BONE_MID,    _C_BONE_SHADOW,  3,  6),
    "r_hip":      (_C_BONE_MID,    _C_BONE_SHADOW,  9, 18),
    "r_knee":     (_C_BONE_BRIGHT, _C_BONE_SHADOW,  7, 14),
    "r_ankle":    (_C_BONE_MID,    _C_BONE_SHADOW,  4,  9),
    "r_toe":      (_C_BONE_MID,    _C_BONE_SHADOW,  3,  6),
}

# ──────────────────────────────────────────────────────────────────────────────
# Action metadata
# ──────────────────────────────────────────────────────────────────────────────
_ACTION_ACCENTS = {
    "run": (255,80,40), "walk": (80,200,120), "jump": (120,200,255),
    "skip": (255,190,80), "wave": (255,220,60), "dance": (220,80,255),
    "fight": (255,50,50), "kick": (255,80,80), "boxing": (255,120,40),
    "fall": (180,140,255), "reach": (80,220,200), "turn": (180,180,255),
    "sit": (100,200,255), "climb": (255,160,60), "swim": (50,170,255),
    "sneak": (100,255,160), "throw": (255,120,60), "idle": (160,160,200),
    "squat": (100,230,180), "lunge": (200,255,100), "stretch": (255,200,100),
    "bow": (200,180,140), "think": (200,150,250), "spin": (255,180,255),
    "clap": (255,230,80), "point": (120,240,255), "shrug": (200,200,200),
    "salute": (150,180,255), "pray": (220,200,255), "hug_self": (255,160,200),
    "carry": (200,160,100), "cheer": (255,200,80), "limp": (200,180,100),
    "crawl": (180,140,100), "roll": (180,220,255), "slide": (140,200,255),
    "eat": (255,150,50), "sleep": (80,80,180), "drink": (50,200,255),
    "read": (220,220,180), "phone": (100,220,100), "push_up": (255,100,80),
    "sit_up": (100,255,180), "push": (255,100,50), "pull": (50,100,255),
}

_ACTION_LABELS = {
    "run": "🏃 Running", "walk": "🚶 Walking", "jump": "🦘 Jumping",
    "skip": "😊 Skipping", "wave": "👋 Waving", "dance": "💃 Dancing",
    "fight": "🥊 Fighting", "kick": "🦵 Kicking", "boxing": "🥊 Boxing",
    "fall": "😱 Falling", "reach": "🤲 Reaching", "turn": "🔄 Turning",
    "sit": "🪑 Sitting", "climb": "🧗 Climbing", "swim": "🏊 Swimming",
    "sneak": "🕵️ Sneaking", "throw": "🎯 Throwing", "idle": "🧍 Standing",
    "squat": "🏋️ Squatting", "lunge": "🤸 Lunging", "stretch": "🧘 Stretching",
    "bow": "🙇 Bowing", "think": "🤔 Thinking", "spin": "🌀 Spinning",
    "clap": "👏 Clapping", "point": "👆 Pointing", "shrug": "🤷 Shrugging",
    "salute": "🫡 Saluting", "pray": "🙏 Praying", "hug_self": "🤗 Hugging",
    "carry": "📦 Carrying", "cheer": "🎉 Cheering", "limp": "🩹 Limping",
    "crawl": "🐾 Crawling", "roll": "🔁 Rolling", "slide": "⛷️ Sliding",
    "eat": "🍔 Eating", "sleep": "💤 Sleeping", "drink": "🥤 Drinking",
    "read": "📖 Reading", "phone": "📱 Calling", "push_up": "💪 Push-Up",
    "sit_up": "🏋️ Sit-Up", "push": "🤚 Pushing", "pull": "✊ Pulling",
}

# ──────────────────────────────────────────────────────────────────────────────
# Rendering Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _draw_bg(draw, w: int, h: int, ground_y: int) -> None:
    """Dark anatomical theatre background with subtle vignette."""
    for y in range(h):
        f = y / h
        # Warm dark charcoal with slight cool tinge at top
        rv = int(8  + f * 10)
        gv = int(8  + f * 8)
        bv = int(12 + f * 12)
        draw.line([(0, y), (w, y)], fill=(rv, gv, bv))

    # Floor
    draw.rectangle([(0, ground_y), (w, h)], fill=(10, 10, 14))
    # Floor grid (perspective lines)
    for x in range(-w, w * 2, 70):
        draw.line([(x, ground_y), (w // 2, h + 200)], fill=(18, 18, 24), width=1)
    for x in range(0, w, 80):
        draw.line([(x, ground_y), (x, h)], fill=(16, 16, 22), width=1)

    # Ground glow line
    for off, alpha in [(-2, 12), (-1, 25), (0, 50), (1, 25), (2, 12)]:
        v = 100 + alpha
        draw.line([(0, ground_y + off), (w, ground_y + off)], fill=(v//2, v//2, v//3))


def _draw_tapered_bone(draw, p1, p2, bone_col, shadow_col, inner_w, outer_w, depth=False):
    """
    Draw a tapered bone between p1 (proximal/wide end) and p2 (distal/narrow end).
    Layers: ambient occlusion shadow → cortex body → periosteum highlight.
    If depth=True (back limb), colors are darkened for depth perception.
    """
    px, py = p1
    cx, cy = p2
    L = math.hypot(cx - px, cy - py)
    if L < 2:
        return

    f = _DEPTH_DARKEN if depth else 1.0
    bc  = _darken(bone_col,   f)
    sc  = _darken(shadow_col, f)
    mid = _darken(tuple(max(0, c - 25) for c in bone_col), f)
    hl  = _darken(tuple(min(255, c + 38) for c in bone_col), f)

    nx = -(cy - py) / L
    ny =  (cx - px) / L

    # Layer 1: ambient-occlusion shadow (widest)
    ws_a, we_a = outer_w * 1.35, outer_w * 0.75
    draw.polygon([
        (px + nx * ws_a, py + ny * ws_a),
        (cx + nx * we_a, cy + ny * we_a),
        (cx - nx * we_a, cy - ny * we_a),
        (px - nx * ws_a, py - ny * ws_a),
    ], fill=sc)

    # Layer 2: cortex body (mid-tone, tapered)
    ws_m, we_m = inner_w * 1.7, inner_w * 0.9
    draw.polygon([
        (px + nx * ws_m, py + ny * ws_m),
        (cx + nx * we_m, cy + ny * we_m),
        (cx - nx * we_m, cy - ny * we_m),
        (px - nx * ws_m, py - ny * ws_m),
    ], fill=mid)

    # Layer 3: core surface (bright ivory)
    ws_c, we_c = inner_w, inner_w * 0.5
    draw.polygon([
        (px + nx * ws_c, py + ny * ws_c),
        (cx + nx * we_c, cy + ny * we_c),
        (cx - nx * we_c, cy - ny * we_c),
        (px - nx * ws_c, py - ny * ws_c),
    ], fill=bc)

    # Layer 4: specular ridge (1-px bright highlight along top edge)
    draw.line([
        (int(px + nx * ws_c * 0.5), int(py + ny * ws_c * 0.5)),
        (int(cx + nx * we_c * 0.5), int(cy + ny * we_c * 0.5)),
    ], fill=hl, width=1)


def _draw_joint_sphere(draw, jx, jy, r, fill_col, shadow_col, depth=False):
    """Render a joint as a sphere with ambient occlusion, body, specular cap."""
    f = _DEPTH_DARKEN if depth else 1.0
    fc = _darken(fill_col,   f)
    sc = _darken(shadow_col, f)
    hl = _darken(tuple(min(255, c + 55) for c in fill_col), f)
    sh = _darken(tuple(max(0, c - 50) for c in fill_col),   f)

    # AO shadow ring
    draw.ellipse((jx-r-3, jy-r-3, jx+r+3, jy+r+3), fill=sc)
    # Body
    draw.ellipse((jx-r, jy-r, jx+r, jy+r), fill=fc)
    # Specular highlight (top-left quadrant)
    ir = max(1, r - 3)
    draw.ellipse((jx-ir, jy-r+1, jx+1, jy-1), fill=hl)
    # Bottom shadow arc
    draw.arc((jx-r+1, jy+1, jx+r-1, jy+r-1), start=20, end=160, fill=sh, width=2)


def _draw_skull(draw, jx, jy, r, depth=False):
    """Detailed skull: cranial dome + orbital sockets + nasal aperture + mandible."""
    f = _DEPTH_DARKEN if depth else 1.0
    glow   = _darken(_C_BONE_GLOW,   f)
    body   = _darken(_C_BONE_MID,    f)
    bright = _darken(_C_BONE_BRIGHT, f)
    shadow = _darken(_C_BONE_SHADOW, f)
    dark   = _darken(_C_BONE_GLOW,   f)

    # Ambient occlusion halo
    draw.ellipse((jx-r-4, jy-r-4, jx+r+4, jy+r+4), fill=glow)
    # Cranial vault
    draw.ellipse((jx-r, jy-r, jx+r, jy+r), fill=body)
    # Upper dome highlight
    hr = max(4, r - 2)
    draw.ellipse((jx-hr, jy-r+2, jx+hr, jy-3), fill=bright)
    # Cranial outline
    draw.ellipse((jx-r, jy-r, jx+r, jy+r), outline=shadow, width=2)

    # Orbital cavities (eye sockets)
    eye_y = jy + max(1, r // 5)
    eye_off = max(2, r // 3)
    eye_r   = max(2, r // 4)
    for ex in (jx - eye_off, jx + eye_off):
        draw.ellipse((ex-eye_r, eye_y-eye_r, ex+eye_r, eye_y+eye_r), fill=dark)
        # Orbital rim highlight
        draw.arc((ex-eye_r, eye_y-eye_r, ex+eye_r, eye_y+eye_r),
                 start=200, end=320, fill=_darken(bright, f * 0.7), width=1)

    # Nasal aperture
    nose_y = eye_y + max(2, r // 4)
    draw.polygon([
        (jx,     nose_y),
        (jx-2,   nose_y + max(2, r//3)),
        (jx+2,   nose_y + max(2, r//3)),
    ], fill=dark)

    # Mandible line (chin)
    chin_y = jy + r - 3
    draw.arc((jx - max(5, r//2), chin_y - 4, jx + max(5, r//2), chin_y + 4),
             start=0, end=180, fill=shadow, width=1)


# ── Depth set: right-side limbs are "back" ────────────────────────────────────
_BACK_JOINTS = {"r_shoulder", "r_elbow", "r_wrist", "r_hip", "r_knee", "r_ankle", "r_toe"}
_BACK_BONES  = {
    ("spine2","r_shoulder"), ("r_shoulder","r_elbow"), ("r_elbow","r_wrist"),
    ("hips","r_hip"), ("r_hip","r_knee"), ("r_knee","r_ankle"), ("r_ankle","r_toe"),
}


def _draw_bg(draw, w: int, h: int, ground_y: int) -> None:
    """Cinematic dark anatomy-theatre background."""
    for y in range(h):
        f = y / h
        draw.line([(0, y), (w, y)], fill=(int(8+f*10), int(8+f*8), int(12+f*12)))
    draw.rectangle([(0, ground_y), (w, h)], fill=(10, 10, 14))
    for x in range(-w, w*2, 70):
        draw.line([(x, ground_y), (w//2, h+200)], fill=(18, 18, 24), width=1)
    for x in range(0, w, 80):
        draw.line([(x, ground_y), (x, h)], fill=(16, 16, 22), width=1)
    for off, alpha in [(-2,12),(-1,25),(0,50),(1,25),(2,12)]:
        v = 100 + alpha
        draw.line([(0,ground_y+off),(w,ground_y+off)], fill=(v//2,v//2,v//3))


def _wrap_text(text: str, font, max_w: int, draw) -> list:
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


def _draw_speech_bubble(draw, head_pos, canvas_w, canvas_h, text, t, font):
    hx, hy = head_pos
    scale = min(1.0, t / 0.4)
    pulse = int(210 + 30 * math.sin(2 * math.pi * t * 0.8))
    max_bubble_w, pad_x, pad_y, line_h = 290, 18, 12, 22

    lines = _wrap_text(text, font, max_bubble_w - pad_x * 2, draw)
    text_w = max(draw.textbbox((0,0),ln,font=font)[2] for ln in lines)
    text_h = len(lines) * line_h
    bw = int((text_w + pad_x*2) * scale)
    bh = int((text_h + pad_y*2) * scale)
    if bw < 6 or bh < 6:
        return

    bx = min(canvas_w - bw - 12, max(12, hx - bw//2 + 40))
    by = max(10, hy - bh - 50)
    tail_x, tail_y = hx + 6, hy - 10

    # Shadow
    draw.rounded_rectangle((bx+4,by+4,bx+bw+4,by+bh+4), radius=14, fill=(4,4,14))
    # Fill
    draw.rounded_rectangle((bx,by,bx+bw,by+bh), radius=14, fill=(16,20,46))
    # Border
    draw.rounded_rectangle((bx,by,bx+bw,by+bh), radius=14, outline=(80,130,pulse), width=2)

    if scale >= 0.8:
        bx_mid = bx + bw//2
        draw.polygon([(bx_mid-8,by+bh),(bx_mid+8,by+bh),(tail_x,tail_y)], fill=(16,20,46))
        draw.line([(bx_mid-8,by+bh),(tail_x,tail_y)], fill=(80,130,pulse), width=2)
        draw.line([(bx_mid+8,by+bh),(tail_x,tail_y)], fill=(80,130,pulse), width=2)

    ty = by + pad_y
    for line in lines:
        lw = draw.textbbox((0,0),line,font=font)[2]
        draw.text((bx+(bw-lw)//2, ty), line, fill=(220,230,255), font=font)
        ty += line_h


def _draw_action_badge(draw, action, canvas_w, canvas_h, t, font):
    accent = _ACTION_ACCENTS.get(action, (200,200,200))
    label  = _ACTION_LABELS.get(action, action.capitalize())
    scale  = min(1.0, t/0.35) if t < 0.35 else 1.0
    pulse  = int(80 + 40 * math.sin(2 * math.pi * t * 0.6))

    bbox = draw.textbbox((0,0), label, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    pad_x, pad_y = 14, 8
    bw = int((tw + pad_x*2) * scale)
    bh = int((th + pad_y*2) * scale)
    bx, by = 14, 14
    if bw < 8 or bh < 8:
        return

    rad = min(14, bw//2, bh//2)
    glow = tuple(c//5 for c in accent)
    draw.rounded_rectangle((bx-3,by-3,bx+bw+3,by+bh+3), radius=min(16,bw//2,bh//2), fill=glow)
    draw.rounded_rectangle((bx,by,bx+bw,by+bh), radius=rad, fill=(12,14,34))
    border = tuple(min(255,c+pulse//4) for c in accent)
    draw.rounded_rectangle((bx,by,bx+bw,by+bh), radius=rad, outline=border, width=2)
    if bh > 16:
        bar_y0, bar_y1 = by+6, by+bh-6
        if bar_y1 > bar_y0:
            draw.rounded_rectangle((bx,bar_y0,bx+4,bar_y1), radius=2, fill=accent)
    draw.text((bx+10+2, by+max(0,(bh-th)//2)), label, fill=accent, font=font)


# ── Action-specific particle effects ─────────────────────────────────────────
def _draw_speed_lines(draw, action, jpos, t, canvas_w, canvas_h):
    if action not in ("run","fight","throw","kick","boxing"):
        return
    accent = _ACTION_ACCENTS.get(action,(255,80,40))
    cx = jpos.get("hips",(canvas_w//2,0))[0]
    cy_ref = jpos.get("spine1",(0,canvas_h//2))[1]
    rnd = np.random.default_rng(seed=int(t*100)%1000)
    for _ in range(9):
        y  = cy_ref + rnd.integers(-110,110)
        x1 = max(0, cx - rnd.integers(80,210))
        x2 = cx - rnd.integers(10,45)
        a  = rnd.integers(35,100)
        col = tuple(int(c*a/255) for c in accent)
        if x1 < x2:
            draw.line([(x1,y),(x2,y)], fill=col, width=max(1,rnd.integers(1,3)))


def _draw_footstep_dust(draw, action, jpos, t, ground_y):
    if action not in ("run","jump","fight","walk","skip","kick","boxing"):
        return
    accent = _ACTION_ACCENTS.get(action,(200,150,100))
    for side in ("l","r"):
        ax,ay = jpos.get(f"{side}_toe", jpos.get(f"{side}_ankle",(0,ground_y)))
        if ay < ground_y - 22:
            continue
        rnd = np.random.default_rng(seed=int(t*100+(0 if side=="l" else 50))%9999)
        for _ in range(5):
            dx = rnd.integers(-24,24)
            dy = rnd.integers(-14,4)
            r  = rnd.integers(3,10)
            a  = rnd.integers(25,85)
            col = tuple(int(c*a/255) for c in accent)
            draw.ellipse((ax+dx-r,ground_y+dy-r,ax+dx+r,ground_y+dy+r), fill=col)


def _draw_energy_ring(draw, action, jpos, t):
    if action not in ("dance","fight","jump","cheer","spin"):
        return
    accent = _ACTION_ACCENTS.get(action,(200,200,200))
    hx,hy = jpos.get("hips",(0,0))
    hy -= 65
    freq = 2.0 if action in ("fight","boxing") else 1.2
    pulse_r = int(85 + 28*math.sin(2*math.pi*t*freq))
    a = int(38+22*abs(math.sin(2*math.pi*t)))
    col = tuple(int(c*a/255) for c in accent)
    draw.ellipse((hx-pulse_r,hy-pulse_r//2,hx+pulse_r,hy+pulse_r//2), outline=col, width=2)


def _draw_swim_bubbles(draw, jpos, t):
    hx,hy = jpos.get("head",(360,200))
    rnd = np.random.default_rng(seed=int(t*30)%300)
    for _ in range(6):
        bx = hx+rnd.integers(-55,55)
        by = hy-rnd.integers(5,65)
        r  = rnd.integers(3,9)
        draw.ellipse((bx-r,by-r,bx+r,by+r), outline=(120,200,255), width=1)


def _draw_stars(draw, jpos, t):
    hx,hy = jpos.get("hips",(360,300))
    rnd = np.random.default_rng(seed=int(t*20)%200)
    for _ in range(7):
        sx = hx+rnd.integers(-110,110)
        sy = hy+rnd.integers(-175,25)
        r  = rnd.integers(2,5)
        col = (rnd.integers(180,255),rnd.integers(180,255),rnd.integers(50,255))
        draw.line([(sx-r*2,sy),(sx+r*2,sy)], fill=col, width=1)
        draw.line([(sx,sy-r*2),(sx,sy+r*2)], fill=col, width=1)
        draw.ellipse((sx-1,sy-1,sx+1,sy+1), fill=col)


def _draw_prayer_glow(draw, jpos, t):
    hx,hy = jpos.get("hips",(360,300))
    hy -= 80
    pulse = int(50+20*abs(math.sin(2*math.pi*t*0.5)))
    col = tuple(int(c*pulse//255) for c in (220,200,255))
    r = 55
    draw.ellipse((hx-r,hy-r//2,hx+r,hy+r//2), outline=col, width=1)


def _draw_direction_arrow(draw, action, jpos, canvas_w):
    if action not in ("walk","run","sneak","limp","skip"):
        return
    accent = _ACTION_ACCENTS.get(action,(200,200,200))
    hx,hy = jpos.get("hips",(canvas_w//2,300))
    ax,ay = hx+62, hy
    draw.line([(ax-32,ay),(ax,ay)], fill=accent, width=3)
    draw.polygon([(ax,ay),(ax-10,ay-6),(ax-10,ay+6)], fill=accent)


def _draw_shadow_ellipse(draw, hx, ground_y):
    shadow_rx = 42
    for rx_off,col in [(12,(14,14,36)),(6,(20,18,44)),(0,(28,24,58))]:
        draw.ellipse((hx-shadow_rx+rx_off,ground_y,hx+shadow_rx-rx_off,ground_y+11), fill=col)


# ──────────────────────────────────────────────────────────────────────────────
# Main Frame Renderer
# ──────────────────────────────────────────────────────────────────────────────
def _render_frame(pose, canvas_w, canvas_h, ground_y, prompt="", t=0.0, action="idle"):
    from PIL import Image, ImageDraw, ImageFont

    img  = Image.new("RGB", (canvas_w, canvas_h), (6,6,20))
    draw = ImageDraw.Draw(img)

    _draw_bg(draw, canvas_w, canvas_h, ground_y)
    jpos = _compute_joints(pose, canvas_w, canvas_h, ground_y)
    hx, hy = jpos.get("hips", (canvas_w//2, ground_y))

    # Pre-effects
    _draw_speed_lines(draw, action, jpos, t, canvas_w, canvas_h)
    _draw_energy_ring(draw, action, jpos, t)

    # Ground shadow
    _draw_shadow_ellipse(draw, hx, ground_y)
    _draw_footstep_dust(draw, action, jpos, t, ground_y)

    # ── Depth-sorted bone rendering ────────────────────────────────────────
    for parent, child in _BONE_DEPTH_ORDER:
        if parent not in jpos or child not in jpos:
            continue
        is_back = (parent, child) in _BACK_BONES
        px,py = jpos[parent]
        cx,cy = jpos[child]
        style = _BONE_STYLES.get(child, (_C_BONE_MID, _C_BONE_SHADOW, 5, 11))
        bone_col, shadow_col, inner_w, outer_w = style
        _draw_tapered_bone(draw, (px,py), (cx,cy), bone_col, shadow_col, inner_w, outer_w, depth=is_back)

    # ── Depth-sorted joint rendering ─────────────────────────────────────
    # Back joints first
    for jname, (jx,jy) in jpos.items():
        if jname not in _BACK_JOINTS:
            continue
        if jname == "head":
            continue
        style = _JOINT_STYLES.get(jname, (_C_BONE_MID, _C_BONE_GLOW, 5))
        fill_col, shadow_col, r = style
        _draw_joint_sphere(draw, jx, jy, r, fill_col, shadow_col, depth=True)

    # Front joints
    for jname, (jx,jy) in jpos.items():
        if jname in _BACK_JOINTS:
            continue
        style = _JOINT_STYLES.get(jname, (_C_BONE_MID, _C_BONE_GLOW, 5))
        fill_col, shadow_col, r = style
        if jname == "head":
            _draw_skull(draw, jx, jy, r)
        else:
            _draw_joint_sphere(draw, jx, jy, r, fill_col, shadow_col, depth=False)

    # Post-effects
    if action == "swim":
        _draw_swim_bubbles(draw, jpos, t)
    if action == "dance":
        _draw_stars(draw, jpos, t)
    if action == "pray":
        _draw_prayer_glow(draw, jpos, t)
    _draw_direction_arrow(draw, action, jpos, canvas_w)

    # Fonts
    try:
        font_badge  = ImageFont.truetype("arial.ttf", 17)
        font_bubble = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font_badge  = ImageFont.load_default()
        font_bubble = font_badge

    _draw_action_badge(draw, action, canvas_w, canvas_h, t, font_badge)
    head_pos = jpos.get("head", (canvas_w//2, 80))
    if prompt:
        _draw_speech_bubble(draw, head_pos, canvas_w, canvas_h, prompt, t, font_bubble)

    return np.array(img)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
_LOOP_ACTIONS = {
    "walk","run","skip","dance","fight","boxing","kick","climb","swim","sneak",
    "idle","wave","drink","eat","read","phone","limp","crawl","slide","spin",
    "clap","point","shrug","salute","pray","hug_self","carry","cheer","push_up",
    "sit_up","push","pull","bow","think","stretch","turn","sit","squat","lunge",
}

def create_animation(hint: str, output_dir: str = "data/generated",
                     fps: int = 30, max_duration: float = 20.0) -> str:
    """
    Generate a realistic procedural skeleton animation for the given text prompt.

    Parameters
    ----------
    hint        : Natural-language description of the desired action.
    output_dir  : Directory to place the output MP4.
    fps         : Frames per second (default 30).
    max_duration: Maximum clip duration in seconds (capped at 30 s).
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = sanitize_hint(hint) + ".mp4"
    out_path = os.path.join(output_dir, filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    actions  = _classify_actions(hint)
    primary  = actions[0][0]
    duration = min(float(max_duration), 30.0)

    if primary in _LOOP_ACTIONS and len(actions) == 1:
        duration = min(duration, 8.0)

    canvas_w, canvas_h = 720, 540
    ground_y = canvas_h - 65

    times  = np.linspace(0, duration, int(duration * fps))
    frames = []

    print(f"[animation_generator] actions={actions} primary={primary} "
          f"duration={duration:.1f}s frames={len(times)}")

    for t in times:
        pose  = _get_pose(primary, float(t), duration, actions=actions)
        frame = _render_frame(pose, canvas_w, canvas_h, ground_y,
                              prompt=hint, t=float(t), action=primary)
        frames.append(frame)

    # Assemble
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

    full_duration = min(float(max_duration), 30.0)
    if clip.duration < full_duration and primary in _LOOP_ACTIONS and len(actions) == 1:
        try:
            from moviepy import concatenate_videoclips
        except ImportError:
            from moviepy.editor import concatenate_videoclips
        reps = math.ceil(full_duration / clip.duration)
        clip = concatenate_videoclips([clip] * reps).subclipped(0, full_duration)

    clip.write_videofile(out_path, codec="libx264", audio=False, logger=None)
    clip.close()
    print(f"[animation_generator] Written → {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a realistic skeleton animation.")
    parser.add_argument("hint", help="Text prompt describing the animation")
    parser.add_argument("--output-dir", "-o", default="data/generated")
    parser.add_argument("--duration",   "-d", type=float, default=10.0)
    args = parser.parse_args()
    out  = create_animation(args.hint, args.output_dir, max_duration=args.duration)
    print(f"animation generated: {out}")