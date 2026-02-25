import os
import subprocess
from animation_generator import create_animation

p = create_animation("diagnostic_test3")
print("created", p)
print("size", os.path.getsize(p))
try:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", p], stderr=subprocess.STDOUT, text=True)
    print("ffprobe output:\n", out)
except Exception as e:
    print("ffprobe unavailable", e)
