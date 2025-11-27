"""
Simple model registry to track trained models and their metrics.
Stores a JSON file at artifacts/model_registry.json with entries like:
{
  "models": [
    {
      "id": 1,
      "timestamp": "2025-11-27T06:40:00Z",
      "path": "models/trained_model.h5",
      "metrics": {"accuracy": 0.95, ...},
      "config": { ... },
      "dataset": {"path": "data/fault_dataset.npz"}
    }
  ]
}
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


REGISTRY_PATH = Path('artifacts/model_registry.json')
METRICS_LATEST = Path('artifacts/metrics_latest.json')


def _load_registry() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text())
        except Exception:
            pass
    return {"models": []}


def _save_registry(reg: Dict[str, Any]):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


def write_latest_metrics(metrics: Dict[str, Any]):
    METRICS_LATEST.parent.mkdir(parents=True, exist_ok=True)
    METRICS_LATEST.write_text(json.dumps(metrics, indent=2))


def register_model(model_path: str, metrics: Dict[str, Any], config: Dict[str, Any] | None = None, dataset_info: Dict[str, Any] | None = None) -> Dict[str, Any]:
    reg = _load_registry()
    next_id = (reg['models'][-1]['id'] + 1) if reg['models'] else 1
    entry = {
        'id': next_id,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'path': model_path,
        'metrics': metrics,
        'config': config or {},
        'dataset': dataset_info or {}
    }
    reg['models'].append(entry)
    _save_registry(reg)
    return entry
