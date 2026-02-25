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
    from moviepy.editor import TextClip, ColorClip, CompositeVideoClip
except ImportError:  # pragma: no cover - just in case the dependency is missing
    TextClip = None
    ColorClip = None
    CompositeVideoClip = None


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



def create_animation(hint: str, output_dir: str = "data/generated") -> str:
    """Generate a simple MP4 animation file for a given textual hint.

    The resulting file is placed in ``output_dir`` using a sanitized version
    of ``hint`` as the basename.  If the file already exists, the existing path
    is returned unchanged.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filename = sanitize_hint(hint) + ".mp4"
    path = os.path.join(output_dir, filename)

    if os.path.exists(path):
        return path

    # fallback: if moviepy isn't available just touch an empty file
    if TextClip is None:
        open(path, "wb").close()
        return path

    # create a 2‑second clip showing the hint text
    txt_clip = TextClip(hint, fontsize=48, color="white", size=(640, 480), method="label")
    bg = ColorClip(size=txt_clip.size, color=(0, 0, 0), duration=2)
    clip = CompositeVideoClip([bg, txt_clip.set_position("center")])
    clip = clip.set_duration(2)
    # Write video quietly; moviepy prints a lot of info otherwise
    clip.write_videofile(path, fps=24, codec="libx264", verbose=False, logger=None)
    clip.close()

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
        help="directory where generated video will be stored",
    )
    args = parser.parse_args()
    out = create_animation(args.hint, args.output_dir)
    print(f"animation generated: {out}")
