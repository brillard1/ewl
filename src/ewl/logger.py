"""
EWL per-sample logger.

Dumps per-sample training data to parquet files for offline analysis in Jupyter.

Each row: step, epoch, sample_id, loss, loss_ema, loss_velocity, signal, weight

Enables three key figures:
  Fig 1 — 2D scatter: loss_magnitude vs loss_velocity, coloured by weight
  Fig 2 — Weight evolution: weight over time for easy / frontier / noisy samples
  Fig 3 — (from multi-run results) rank vs Δaccuracy

Usage:
    logger = EWLLogger("outputs/my_run/ewl_log")

    # inside training loop, after compute_weights:
    logger.log_step(global_step, epoch, ema_weighter)

    # at end of training:
    logger.finish()

    # in notebook:
    df = EWLLogger.load("outputs/my_run/ewl_log")
"""

import numpy as np
from pathlib import Path


class EWLLogger:

    def __init__(self, log_dir, flush_every=2000):
        """
        Args:
            log_dir:     Directory to write parquet chunks into.
            flush_every: Flush buffer to disk after this many rows.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._buffer = []
        self._chunk_idx = 0

    # ------------------------------------------------------------------

    def log_step(self, step, epoch, weighter):
        """
        Record the last batch processed by weighter.
        Call once per forward pass, after compute_weights().
        """
        batch = weighter._last_batch
        if batch is None:
            return

        ids     = batch["ids"]
        losses  = batch["loss"]
        emas    = batch["loss_ema"]
        prevs   = batch["prev_ema"]
        signals = batch["signal"]   # None during warmup
        weights = batch["weight"]

        for i in range(len(ids)):
            self._buffer.append({
                "step":           int(step),
                "epoch":          int(epoch),
                "sample_id":      int(ids[i]),
                "loss":           float(losses[i]),
                "loss_ema":       float(emas[i]),
                "loss_velocity":  float(prevs[i] - losses[i]),  # positive = dropping
                "signal":         float(signals[i]) if signals is not None else float("nan"),
                "weight":         float(weights[i]),
            })

        if len(self._buffer) >= self.flush_every:
            self._flush()

    def finish(self):
        """Flush remaining buffer. Call at end of training."""
        if self._buffer:
            self._flush()

    # ------------------------------------------------------------------

    def _flush(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for EWLLogger: pip install pandas pyarrow")

        df = pd.DataFrame(self._buffer)
        path = self.log_dir / f"chunk_{self._chunk_idx:04d}.parquet"
        df.to_parquet(path, index=False)
        self._chunk_idx += 1
        self._buffer = []

    # ------------------------------------------------------------------

    @staticmethod
    def load(log_dir):
        """
        Load all parquet chunks from log_dir into a single DataFrame.

        Returns:
            pd.DataFrame with columns:
                step, epoch, sample_id, loss, loss_ema,
                loss_velocity, signal, weight
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required: pip install pandas pyarrow")

        files = sorted(Path(log_dir).glob("chunk_*.parquet"))
        if not files:
            raise FileNotFoundError(f"No log chunks found in {log_dir}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
