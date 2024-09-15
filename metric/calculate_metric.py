from typing import Dict, Optional

from .overlap import overlap_metrics
from .registration import pairwise_metrics, registration_metrics

def compute_metrics(predictions: Dict, ground_truth: Dict, info: Optional[Dict] = None) -> Dict:
    metrics = {}

    if "overlap_pred" in predictions and "overlap_gt" in ground_truth:
        metrics.update(overlap_metrics(predictions, ground_truth, info))
    
    if "transform_pred" in predictions and "transform_gt" in ground_truth:
        metrics.update(pairwise_metrics(predictions, ground_truth))

    if "abs_rotation_pred" in predictions and "abs_translation_pred" in predictions and info is not None:
        metrics.update(registration_metrics(predictions, ground_truth, info))

    return metrics