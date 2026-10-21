"""
Performance monitoring with DELAYED labels.

- Predictions are stored when a window is served.
- The true labels of window w are released when window w + label_delay arrives.
- Accuracy / F1 / per-class recall are computed over the last `performance_windows`
  labelled windows and compared with the baseline.
- ADWIN (from the `river` library) watches the stream of 0/1 errors and raises an
  alarm when the recent error rate is significantly different from the older one.
"""
import numpy as np
from river.drift import ADWIN

from src.c4.monitoring.stats import classification_summary


class PerformanceTracker:
    def __init__(self, label_delay: int, performance_windows: int, adwin_delta: float,
                 adwin_recent_windows: int):
        self.delay = label_delay
        self.k = performance_windows
        self.adwin = ADWIN(delta=adwin_delta)
        self.adwin_recent = adwin_recent_windows
        self.served = {}             # window -> (y_true, y_pred)
        self.labelled = []           # windows whose labels have arrived, in order
        self.last_adwin_alarm = None  # window index (decision time) of the last ADWIN alarm

    def add_served(self, window: int, y_true, y_pred) -> None:
        self.served[window] = (np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int))

    def release_labels(self, current_window: int) -> None:
        """Labels for (current_window - delay) arrive now; feed their errors to ADWIN."""
        w = current_window - self.delay
        if w < 0 or w not in self.served or w in self.labelled:
            return
        self.labelled.append(w)
        y, p = self.served[w]
        for err in (y != p).astype(int):
            self.adwin.update(int(err))
            if self.adwin.drift_detected:
                self.last_adwin_alarm = current_window

    def evaluate(self, current_window: int, baseline: dict) -> dict:
        """Metrics over the last k labelled windows, compared with the baseline."""
        self.release_labels(current_window)
        if len(self.labelled) < self.k:      # wait until k labelled windows exist (less noise)
            return {"available": False}
        recent = self.labelled[-self.k:]
        y = np.concatenate([self.served[w][0] for w in recent])
        p = np.concatenate([self.served[w][1] for w in recent])
        m = classification_summary(y, p)
        b = baseline["performance"]
        acc_drop = b["accuracy"] - m["accuracy"]
        f1_drop = b["f1"] - m["f1"]
        adwin_recent = (self.last_adwin_alarm is not None and
                        current_window - self.last_adwin_alarm < self.adwin_recent)
        return {
            "available": True,
            "labelled_windows": recent,
            "latest_labelled_window": recent[-1],
            "earliest_labelled_window": recent[0],
            **m,
            "accuracy_drop": float(acc_drop),
            "f1_drop": float(f1_drop),
            "perf_drop": float(max(acc_drop, f1_drop)),
            "label_share_delta": float(m["positive_share"] - baseline["label"]["positive_share"]),
            "tpr_delta": float(m["tpr"] - b["tpr"]),
            "tnr_delta": float(m["tnr"] - b["tnr"]),
            "adwin_alarm": bool(adwin_recent),
        }
