"""Live training monitor. Run: python watch_training.py"""
import json, time, numpy as np

PATH = r"D:\WeSaveTime-ASEAN\checkpoints\training_log.json"
while True:
    try:
        d = json.load(open(PATH))
        eps = d["episodes"]
        e = eps[-1]
        best = max(x["mean_reward"] for x in eps)
        print(
            f'Ep {e["episode"]}/{e["total_episodes"]} | '
            f'R={e["mean_reward"]:+.3f} | '
            f'Loss={e["mean_loss"]:.4f} | '
            f'Wait={e.get("avg_wait",0):.1f}s | '
            f'Best={best:+.3f} | '
            f'[{len(eps)}/500]'
        )
    except Exception:
        print("Waiting for log...")
    time.sleep(15)
