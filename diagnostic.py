import os
import subprocess
from animation_generator import create_animation

# try a few different hints so we can manually inspect the output files
for hint in (
    "diagnostic_test3",
    "stick figure waving hello",
    "running in the park",
    "jumping high",
    "I love you",
    "hey there",
    "let's dance",
):
    # use default directory (data/animations) so files appear with the
    # existing library when you inspect it later
    p = create_animation(hint)
    print("hint ->", hint)
    print("  created", p)
    print("  size", os.path.getsize(p))
    try:
        out = subprocess.check_output([
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            p,
        ], stderr=subprocess.STDOUT, text=True)
        print("  ffprobe output:\n", out)
    except Exception as e:
        print("  ffprobe unavailable", e)
    print()
