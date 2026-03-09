from animation_generator import create_animation
import os

prompts = [
    ("I need to run", "run"),
    ("wave hello", "wave"),
    ("jump higher", "jump"),
    ("dancing happily", "dance"),
]

for prompt, tag in prompts:
    out_f = f"data/generated/test_{tag}.mp4"
    if os.path.exists(out_f):
        os.remove(out_f)
    try:
        p = create_animation(prompt, "data/generated", max_duration=10.0)
        size = os.path.getsize(p) if os.path.exists(p) else 0
        print(f"[{tag}] OK  size={size:,} bytes  path={p}")
    except Exception as e:
        import traceback
        print(f"[{tag}] FAIL: {e}")
        traceback.print_exc()
