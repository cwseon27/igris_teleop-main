import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

class LossLogger:
    def __init__(self, log_dir: Path, run_name: str = "train", keep_last: int = 5000):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = log_dir / f"{run_name}_loss.jsonl"
        # line-buffered: write() 호출마다 줄 단위로 OS에 밀어넣음(너무 자주 flush 하지 않아도 됨)
        self._fp = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        self._cache = deque(maxlen=keep_last)  # 플롯용(최근 N개만 메모리 유지)
        self._has_header = False

    def log(self, epoch: int, loss: float, lr: Optional[float] = None, split: str = "train"):
        rec = {
            "t": time.time(),
            "epoch": int(epoch),
            "loss": float(loss),
            "lr": None if lr is None else float(lr),
            "split": split,
        }
        self._fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._cache.append(rec)

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass

    def save_plot(self, png_path: Path, x_key: str = "epoch", split: str = "train", y_lim=None):
        xs, ys = [], []
        for r in self._cache:
            if r["split"] != split:
                continue
            xs.append(r[x_key])
            ys.append(r["loss"])
        if not xs:
            return
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys, label=f"{split}/loss")
        plt.title("Loss over Epochs (recent)")
        plt.xlabel(x_key)
        plt.ylabel("loss")
        plt.grid(True)
        plt.legend()
        if y_lim is not None:
            plt.ylim(*y_lim)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

    def export_csv(self, csv_path: Path):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["t", "epoch", "loss", "lr", "split"]
        with self.jsonl_path.open("r", encoding="utf-8") as fin, csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as fout:
            w = csv.DictWriter(fout, fieldnames=fieldnames)
            w.writeheader()
            for line in fin:
                r = json.loads(line)
                w.writerow({k: r.get(k, None) for k in fieldnames})