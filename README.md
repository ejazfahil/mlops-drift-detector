# mlops-drift-detector — statistical drift monitoring for production ML

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)
![Status](https://img.shields.io/badge/status-core%20implemented-success)

A lightweight, dependency-minimal toolkit for detecting **data drift** and
**concept drift** in deployed models, and exporting the signals to Prometheus so
they become first-class observability metrics with alerting.

---

## Overview & Aim

Models decay silently. The input distribution shifts, the relationship between
features and target drifts, and aggregate accuracy — if it is even measurable in
production — moves too slowly to act on. This project treats drift as a
**monitoring problem**: compute principled, well-understood statistics on
incoming data and surface them as Prometheus gauges that an SRE/ML team can alert
and dashboard on, exactly like any other production signal.

The design separates three concerns:

| Concern | Question answered | Implementation |
|---|---|---|
| **Feature drift** | Has the *input* distribution moved? | Population Stability Index (PSI) per numeric feature |
| **Concept drift** | Has the *error stream* changed regime? | Page–Hinkley sequential change-point test |
| **Streaming** | Detect drift online, one sample at a time | Bounded-window monitor with reset + callback |
| **Export** | Make it observable | Prometheus text-exposition exporter |

---

## Methodology / How It Works

### 1. Feature drift — Population Stability Index (PSI)

For a reference (training) distribution and a current (production) window, each
numeric feature is binned into `n_bins` (default 10) fixed-edge buckets derived
from the reference range. With reference proportions $e_i$ and current
proportions $a_i$ in bin $i$, PSI is the **symmetric population-stability sum**:

$$\mathrm{PSI} = \sum_{i=1}^{B} (a_i - e_i)\,\ln\!\frac{a_i}{e_i}$$

Bin proportions are floored at $\epsilon = 10^{-6}$ to keep the log finite, and a
degenerate (zero-variance) feature returns 0. The conventional banding is applied
verbatim in [`src/metrics/psi.py`](src/metrics/psi.py):

| PSI | Interpretation |
|---|---|
| `< 0.10` | `stable` |
| `0.10 – 0.25` | `moderate_shift` |
| `> 0.25` | `significant_shift` |

[`FeatureDriftDetector`](src/detectors/feature_drift.py) caches the reference
arrays per numeric column at construction, then on each `detect(current)` call
returns a per-feature `{psi, status}` map, the list of drifted columns (those
above the threshold, default `0.25`), and an `overall` boolean.

### 2. Concept drift — Page–Hinkley test

[`PageHinkleyDetector`](src/detectors/concept_drift.py) is a classic sequential
change-point detector run over a stream of error/loss values. It maintains a
running mean $\bar{x}_t$ (EW-updated by `alpha`), accumulates the magnitude-and-
direction sum

$$m_t = \sum_{k\le t}\bigl(x_k - \bar{x}_k - \delta\bigr),\qquad
M_t = \min_{k\le t} m_k,$$

and flags drift when the deviation $m_t - M_t$ exceeds a threshold $\lambda$
(default `50.0`), with `delta` a tolerance slack. This catches a sustained upward
shift in the error stream without storing the full history.

### 3. Streaming monitor

[`StreamingDriftMonitor`](src/detectors/streaming_detector.py) wraps the
Page–Hinkley detector with a bounded `deque` window, a drift counter, and an
`on_drift` callback. On a positive detection it fires the callback (carrying the
step index and cumulative drift count) and **auto-resets** the detector so it can
catch the next regime change.

```
errors ──▶ StreamingDriftMonitor.update(err)
              │  append to window (maxlen=window)
              ▼
        PageHinkleyDetector.update(err) ──▶ drift?
              │ yes                              │ no
              ▼                                  ▼
        on_drift({t, n}); ph.reset()        return False
```

### 4. Prometheus export

[`PrometheusExporter`](src/exporters/prometheus.py) renders a results dict into
the Prometheus text-exposition format:

```
# TYPE ml_drift_feature_psi gauge
ml_drift_feature_psi{feature="amount"} 0.31
ml_drift_overall_rate 0.14
```

These scrape cleanly into Prometheus and drive alert rules
(see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)).

---

## Tech Stack & Tools

- **Python 3.11+**
- **NumPy** — histogramming, PSI computation
- **pandas** — feature-frame handling and dtype selection
- **Prometheus** — metric export (text exposition format) + alerting
- **Docker** — containerized monitor (see deployment guide)
- Standard library only for the streaming path (`collections.deque`, `dataclasses`)

---

## Project Structure

```
mlops-drift-detector/
├── src/
│   ├── metrics/
│   │   └── psi.py                  # compute_psi() + interpret_psi() banding
│   ├── detectors/
│   │   ├── feature_drift.py        # FeatureDriftDetector (PSI per numeric feature)
│   │   ├── concept_drift.py        # PageHinkleyDetector (sequential change point)
│   │   └── streaming_detector.py   # StreamingDriftMonitor (online + callback)
│   └── exporters/
│       └── prometheus.py           # PrometheusExporter (text exposition format)
├── tests/
│   └── test_detectors.py           # no-drift vs drift assertions (seeded Normals)
└── docs/
    └── DEPLOYMENT.md               # Dockerfile + Prometheus alert rule
```

---

## Key Features

- **PSI feature drift** with the standard stable/moderate/significant banding.
- **Page–Hinkley concept drift** for online error-stream change detection.
- **Streaming monitor** with bounded memory, auto-reset, and a drift callback.
- **Prometheus-native export** — drift becomes a gauge you can alert on.
- **Minimal dependencies** — NumPy/pandas for the batch path; stdlib for streaming.
- **Tested behavior** — `tests/test_detectors.py` verifies that two samples from
  the same Normal do **not** drift, while a mean shift of +10σ **does**.

---

## Results

This repository ships the **detector library and a behavioral test suite**, not a
benchmark report. The included test (`tests/test_detectors.py`) demonstrates
correct directionality on seeded synthetic data — no-drift inputs return
`overall=False`, a large mean shift returns `overall=True`. No production drift
scores are claimed here; PSI/Page–Hinkley values are data-dependent and computed
at runtime.

---

## Getting Started

```bash
# Run the behavioral tests
python -m pytest tests/ -q
```

```python
import pandas as pd
from src.detectors.feature_drift import FeatureDriftDetector
from src.detectors.streaming_detector import StreamingDriftMonitor
from src.exporters.prometheus import PrometheusExporter

# Batch feature drift
ref = pd.DataFrame({"amount": [...]})
det = FeatureDriftDetector(ref, threshold=0.25)
report = det.detect(current_df)           # {"features": {...}, "drifted": [...], "overall": bool}
print(PrometheusExporter().export(report))

# Online concept drift on an error stream
mon = StreamingDriftMonitor(window=1000, on_drift=lambda e: print("DRIFT", e))
for err in error_stream:
    mon.update(err)
```

Deployment (Docker image + a `HighDrift` Prometheus alert firing on
`ml_drift_feature_psi > 0.25` for 5m) is documented in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Challenges

- **Numerical stability of PSI** — empty/clipped bins and zero-variance features
  must be handled before taking logs; both are guarded with an $\epsilon$ floor.
- **Memory-bounded streaming** — Page–Hinkley keeps drift detection $O(1)$ per
  sample, avoiding unbounded history storage.
- **Avoiding alert storms** — auto-reset after a detection prevents a single
  sustained shift from firing on every subsequent sample.

## Future Work

- Categorical-feature drift (chi-square / Jensen–Shannon) alongside numeric PSI.
- Additional distances (KL, Wasserstein) selectable per feature.
- A scheduled `python -m src.monitor` entrypoint (referenced in the Dockerfile)
  wiring batch detection to a Prometheus `/metrics` endpoint.
- Drift-attribution (which features dominate the overall signal).

## Conclusion

`mlops-drift-detector` packages the statistics that matter for catching silent
model decay — PSI for input drift, Page–Hinkley for concept drift — behind a
small, well-tested API, and exposes them in the one place an operations team will
actually see them: Prometheus.
