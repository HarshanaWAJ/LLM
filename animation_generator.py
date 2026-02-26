import os
import re

# The animation generator module creates simple placeholder videos based on
# textual hints.  It is intentionally very lightweight – the generated
# animations simply display the hint text on a solid background.  This gives
# the rest of the system something to reference until real filmed animations
# are produced.

# moviepy is used for video creation.  If you don't have it installed you can
# add ``moviepy`` to requirements.txt and ``pip install -r requirements.txt``
# (the project already uses ``numpy`` and ``tqdm`` which pull in ``imageio``
# so there are no heavy new dependencies).

try:
    from moviepy import TextClip, ColorClip, CompositeVideoClip, VideoClip
except ImportError:  # pragma: no cover - just in case the dependency is missing
    TextClip = None
    ColorClip = None
    CompositeVideoClip = None
    VideoClip = None

# optional PIL support for drawing stick figures/bubbles
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


VALID_FILENAME = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_hint(hint: str) -> str:
    """Convert an arbitrary text hint into a safe filename.

    Spaces are replaced with underscores and any character not in
    ``[A-Za-z0-9_-]`` is removed.  Multiple underscores are collapsed and the
    string is lowercased.
    """
    cleaned = hint.strip().lower()
    cleaned = cleaned.replace(" ", "_")
    cleaned = VALID_FILENAME.sub("", cleaned)
    # collapse repeated underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned:
        cleaned = "animation"
    return cleaned



def create_animation(hint: str, output_dir: str = "data/generated", template_path: str | None = None) -> str:
    """Generate a simple MP4 animation file for a given textual hint.

    The output file is normally stored in ``output_dir`` (which defaults to
    ``data/animations`` so new clips appear alongside the hand‑crafted
    library).  If ``template_path`` points at an existing video file, the
    generated clip will try to match its resolution, frame rate and duration
    where possible.  This makes it easier to create placeholders that "look
    like" the real animations.

    Older versions of the system simply rendered the hint as white text on a
    black background.  The engine has been enhanced to draw a crude stick
    figure performing a rudimentary animation inferred from the hint (e.g.
    waving, running, jumping).  The prompt itself is shown inside a little
    "thinking" bubble above the figure's head.

    The resulting file is placed in ``output_dir`` using a sanitized version
    of ``hint`` as the basename.  If the file already exists, the existing
    path is returned unchanged.
    """
    # ensure output directory exists; default is the same place where
    # hand‑crafted animations live so new clips blend in visually.
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filename = sanitize_hint(hint) + ".mp4"
    path = os.path.join(output_dir, filename)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    # fallback: if moviepy isn't available just touch an empty file
    if TextClip is None or VideoClip is None:
        open(path, "wb").close()
        return path

    # if a template was provided, try to read its properties
    size = (640, 480)
    duration = 2.0
    fps = 24
    if template_path is not None and os.path.exists(template_path):
        try:
            from moviepy import VideoFileClip as _VFC

            tpl = _VFC(template_path)
            size = tpl.size
            duration = tpl.duration
            fps = tpl.fps or fps
            tpl.close()
        except Exception:
            # ignore problems with reading template
            pass

    # helper utilities ----------------------------------------------------
    def _parse_action(hint_text: str) -> str:
        """Return a simple action keyword based on words in the hint.

        The mapping is intentionally naive; it scans the lowercased text for
        any of a handful of keywords and returns the associated animation.
        Prompts that don't contain one of the known words result in the
        default ``"idle"`` animation.

        You can extend or tweak the ``keyword_map`` to interpret whatever
        prompts make sense for your application.  The order of the list gives
        priority when multiple keywords are present.
        """
        low = hint_text.lower()
        keyword_map: dict[str, list[str]] = {
            "wave": ["wave", "hello", "hi", "hey"],
            "run": ["run", "jog", "walk", "sprint"],
            "jump": ["jump", "hop", "leap"],
            "dance": ["dance", "party", "celebrate", "love", "happy", "yay"],
            # future actions could include "cry", "angry", etc.
        }
        for action, kws in keyword_map.items():
            for kw in kws:
                if kw in low:
                    return action
        return "idle"

    def _get_font(size: int):
        # try to load a truetype font; fall back to default bitmap
        if ImageFont is None:
            return None
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def _wrap_text(text: str, font, max_width: int):
        if font is None:
            return [text]
        words = text.split()
        lines: list[str] = []
        # we need a draw object to measure text width
        if Image is not None and ImageDraw is not None:
            dummy = Image.new("RGB", (1, 1))
            measurer = ImageDraw.Draw(dummy)
        else:
            measurer = None
        while words:
            line = words.pop(0)
            while words:
                candidate = line + " " + words[0]
                if measurer is not None:
                    bbox = measurer.textbbox((0, 0), candidate, font=font)
                    w = bbox[2] - bbox[0]
                else:
                    w = len(candidate) * 10
                if w <= max_width:
                    line = candidate
                    words.pop(0)
                else:
                    break
            lines.append(line)
        return lines

    def _draw_frame(hint_text: str, action: str, size=(640, 480), t_norm=0.0):
        """Return a numpy array representing one frame of the stick figure animation."""
        import numpy as np
        import math

        # create blank background
        if Image is None or ImageDraw is None:
            # fall back to simple text clip behaviour
            img = Image.new("RGB", size, (0, 0, 0))
            draw = ImageDraw.Draw(img)
            f = _get_font(24)
            draw.text((10, 10), hint_text, font=f, fill="white")
            return np.array(img)

        img = Image.new("RGB", size, (0, 0, 0))
        draw = ImageDraw.Draw(img)
        cx, cy = size[0] // 2, size[1] // 2

        # stick figure geometry
        head_r = 20
        body_len = 60
        shoulder_y = cy - head_r - 20 + head_r * 2 + 10
        hip_y = shoulder_y + body_len
        arm_len = 40
        leg_len = 50

        # compute motion parameters
        arm_angle = math.pi / 4
        leg_angle = math.pi / 6
        bounce = 0
        if action == "wave":
            arm_angle = math.pi / 4 + math.sin(2 * math.pi * t_norm) * (math.pi / 6)
        elif action == "run":
            leg_angle = math.sin(2 * math.pi * t_norm) * (math.pi / 6)
            arm_angle = -leg_angle
        elif action == "jump":
            bounce = math.sin(2 * math.pi * t_norm) * 10
        elif action == "dance":
            arm_angle = math.pi / 4 + math.sin(4 * math.pi * t_norm) * (math.pi / 8)
            leg_angle = math.sin(4 * math.pi * t_norm) * (math.pi / 8)

        # draw head
        draw.ellipse(
            (cx - head_r, cy - head_r - 20 + bounce, cx + head_r, cy + head_r - 20 + bounce),
            outline="white",
            width=3,
        )

        # body
        draw.line(
            (cx, cy - head_r - 20 + head_r * 2 + bounce, cx, cy - head_r - 20 + head_r * 2 + body_len + bounce),
            fill="white",
            width=3,
        )

        # arms
        for sign in (1, -1):
            angle = arm_angle if sign == 1 else math.pi - arm_angle
            ax = cx + arm_len * math.cos(angle)
            ay = shoulder_y + arm_len * math.sin(angle) + bounce
            draw.line((cx, shoulder_y + bounce, ax, ay), fill="white", width=3)

        # legs
        for angle in (math.pi / 2 + leg_angle, math.pi / 2 - leg_angle):
            lx = cx + leg_len * math.cos(angle)
            ly = hip_y + leg_len * math.sin(angle) + bounce
            draw.line((cx, hip_y + bounce, lx, ly), fill="white", width=3)

        # thinking bubble
        bubble_left, bubble_top = cx + 80, cy - 180
        bubble_right, bubble_bottom = bubble_left + 220, bubble_top + 100
        draw.ellipse((bubble_left, bubble_top, bubble_right, bubble_bottom), outline="white", width=3)
        # tail dots
        tail_points = [(cx + 40, cy - 120), (cx + 60, cy - 140), (cx + 70, cy - 160)]
        for (x, y) in tail_points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="white", width=2)

        # bubble text
        font = _get_font(24)
        lines = _wrap_text(hint_text, font, bubble_right - bubble_left - 10)
        y_text = bubble_top + 10
        for line in lines:
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), line, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            else:
                w, h = draw.textsize(line, font=font)
            draw.text(
                (bubble_left + (bubble_right - bubble_left - w) / 2, y_text),
                line,
                fill="white",
                font=font,
            )
            y_text += h + 2

        return np.array(img)

    def _make_animation_clip(hint_text: str, duration: float = duration, size=size):
        action = _parse_action(hint_text)

        def frame(t):
            # normalize t to [0,1]
            return _draw_frame(hint_text, action, size=size, t_norm=(t / duration))

        return VideoClip(frame, duration=duration)

    try:
        clip = _make_animation_clip(hint, duration=duration, size=size)
        clip.write_videofile(path, fps=fps, codec="libx264", logger=None)
        clip.close()
    except Exception as e:
        print(f"Error generating animation with moviepy: {e}")
        open(path, "wb").close()

    return path


if __name__ == "__main__":
    # simple command‑line frontage so users can manually create animations
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a placeholder animation for a given text hint."
    )
    parser.add_argument("hint", help="textual hint or prompt to turn into a video")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/generated",
        help="directory where generated video will be stored (defaults to data/generated)",
    )
    args = parser.parse_args()
    out = create_animation(args.hint, args.output_dir)
    print(f"animation generated: {out}")
