import json
from pathlib import Path
import pytest

ARTIFACTS = Path('artifacts')
LATEST = ARTIFACTS / 'metrics_latest.json'
BASELINE = ARTIFACTS / 'baseline.json'


@pytest.mark.skipif(not LATEST.exists(), reason="No latest metrics found; run training to generate.")
def test_metrics_schema_present():
    metrics = json.loads(LATEST.read_text())
    # Basic keys expected
    for k in [
        'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'inference_time_ms'
    ]:
        assert k in metrics, f"Missing metric: {k}"
    assert 0.0 <= metrics['accuracy'] <= 1.0


@pytest.mark.skipif(not (LATEST.exists() and BASELINE.exists()), reason="Baseline or latest metrics missing.")
def test_metrics_meet_baseline():
    latest = json.loads(LATEST.read_text())
    base = json.loads(BASELINE.read_text())
    # Compare weighted F1 with small tolerance
    assert latest.get('f1_weighted', 0.0) + 1e-6 >= base.get('f1_weighted', 0.0), (
        f"F1 dropped: latest {latest.get('f1_weighted')} < baseline {base.get('f1_weighted')}"
    )
