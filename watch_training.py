"""Live training monitor. Run: python watch_training.py"""
import json, time, traceback

PATH = r"D:\WeSaveTime-ASEAN\checkpoints\training_log.json"
while True:
    try:
        d = json.load(open(PATH))
        eps = d["episodes"]
        e = eps[-1]
        total_eps = e.get("total_episodes", len(eps))
        best = max(x["mean_reward"] for x in eps)
        taw = e.get("time_avg_wait", e.get("avg_wait", 0))
        print(
            f'Ep {e["episode"]}/{total_eps} | '
            f'R={e["mean_reward"]:+.3f} | '
            f'Loss={e["mean_loss"]:.4f} | '
            f'Wait={e.get("avg_wait",0):.1f}s | '
            f'TAWait={taw:.1f}s | '
            f'Best={best:+.3f}'
        )
    except (FileNotFoundError, IOError, OSError):
        print("Waiting for log...")
    except Exception:
        traceback.print_exc()
        raise
    time.sleep(15)
